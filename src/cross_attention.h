#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "add.h"
#include "attention.h"
#include "layers.h"
#include "tensor.h"
#include "transformer.h"

namespace zf {

// Multi-head cross attention: queries come from one sequence, keys and values
// from another.
//
// This is the piece self attention cannot express. In self attention every
// position both asks and answers, so the two sequences must be concatenated
// into one and the model is free to attend any which way inside the join. Here
// the asking side and the answering side are separate tensors with separate
// lengths, which is what makes an encoder/decoder split possible: the encoder
// output is read many times and never rewritten, and the decoder holds only as
// many positions as there are things to predict.
//
// `memory_length` masks trailing memory positions, the same padding trick
// MultiHeadSelfAttention uses for its own sequence; the query side has its own
// `valid_length`. Both default to the full compile-time length.
//
// Batch: `Batch` independent examples stacked row-wise. Queries are
// Batch * QueryLen rows and memory Batch * MemoryLen rows; example b's queries
// only ever see example b's memory, each with its own valid and memory length.
// The result is exactly that of Batch separate calls.
template <typename T, std::size_t QueryLen, std::size_t MemoryLen,
          std::size_t ModelDim, std::size_t Batch = 1>
class MultiHeadCrossAttention {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");
  static_assert(Batch > 0, "a batch holds at least one example");

 public:
  using value_type = T;
  static constexpr std::size_t kQueryRows = Batch * QueryLen;
  static constexpr std::size_t kMemoryRows = Batch * MemoryLen;
  using QuerySequence = Tensor<T, kQueryRows, ModelDim>;
  using MemorySequence = Tensor<T, kMemoryRows, ModelDim>;
  // Row = query row (all examples stacked), column = memory slot within its
  // example.
  using Scores = Tensor<T, kQueryRows, MemoryLen>;

  static constexpr std::size_t query_len = QueryLen;
  static constexpr std::size_t memory_len = MemoryLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t batch = Batch;
  static constexpr std::size_t parameter_count =
      4 * Linear<T, ModelDim, ModelDim>::parameter_count;

  explicit MultiHeadCrossAttention(std::size_t heads) : heads_(heads) {
    if (heads == 0 || ModelDim % heads != 0) {
      throw std::invalid_argument(
          "head count " + std::to_string(heads) +
          " must be non-zero and divide model dim " + std::to_string(ModelDim));
    }
    attn_.assign(heads_, Scores{});
    set_valid_length(QueryLen);
    set_memory_length(MemoryLen);
  }

  std::size_t heads() const { return heads_; }
  std::size_t head_dim() const { return ModelDim / heads_; }

  // Example 0's lengths; with Batch == 1, the lengths.
  std::size_t memory_length() const { return memory_spans_[0].count; }
  std::size_t memory_length(std::size_t b) const { return memory_spans_.at(b).count; }
  std::size_t valid_length() const { return query_spans_[0].count; }
  std::size_t valid_length(std::size_t b) const { return query_spans_.at(b).count; }

  // Throws std::out_of_range unless 1 <= length <= MemoryLen. Zero would leave
  // every query with nothing to attend to and no softmax to take. The
  // one-argument forms set every example.
  void set_memory_length(std::size_t length) {
    for (std::size_t b = 0; b < Batch; ++b) set_memory_length(b, length);
  }
  void set_memory_length(std::size_t b, std::size_t length) {
    if (length == 0 || length > MemoryLen) {
      throw std::out_of_range("memory length " + std::to_string(length) +
                              " outside [1, " + std::to_string(MemoryLen) +
                              "]");
    }
    memory_spans_.at(b) = RowSpan{b * MemoryLen, length};
  }

  void set_valid_length(std::size_t length) {
    for (std::size_t b = 0; b < Batch; ++b) set_valid_length(b, length);
  }
  void set_valid_length(std::size_t b, std::size_t length) {
    if (length == 0 || length > QueryLen) {
      throw std::out_of_range("valid length " + std::to_string(length) +
                              " outside [1, " + std::to_string(QueryLen) + "]");
    }
    query_spans_.at(b) = RowSpan{b * QueryLen, length};
  }

  RowSpans row_spans() const { return query_spans_; }
  RowSpans memory_spans() const { return memory_spans_; }

  template <typename Rng>
  void randomize(Rng& rng) {
    wq_.randomize(rng);
    wk_.randomize(rng);
    wv_.randomize(rng);
    wo_.randomize(rng);
  }

  QuerySequence forward(const QuerySequence& q_in,
                        const MemorySequence& memory) {
    const std::size_t hd = head_dim();
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    q_in_ = q_in;
    memory_ = memory;
    // Only the rows that exist: queries past the valid length and memory past
    // the memory length are never read.
    q_.fill(T{});
    k_.fill(T{});
    v_.fill(T{});
    wq_.forward_spans(q_in, q_, query_spans_);
    wk_.forward_spans(memory, k_, memory_spans_);
    wv_.forward_spans(memory, v_, memory_spans_);
    ctx_.fill(T{});

    const T* ZF_RESTRICT qp = q_.data();
    const T* ZF_RESTRICT kp = k_.data();
    const T* ZF_RESTRICT vp = v_.data();
    T* ZF_RESTRICT cp = ctx_.data();

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      T* ZF_RESTRICT ap = attn_[h].data();
      for (std::size_t b = 0; b < Batch; ++b) {
        const std::size_t qbase = b * QueryLen;
        const std::size_t valid = query_spans_[b].count;
        const std::size_t limit = memory_spans_[b].count;
        const T* ZF_RESTRICT kb = kp + b * MemoryLen * ModelDim + off;
        const T* ZF_RESTRICT vb = vp + b * MemoryLen * ModelDim + off;
        for (std::size_t i = 0; i < valid; ++i) {
          const T* ZF_RESTRICT qi = qp + (qbase + i) * ModelDim + off;
          T* ZF_RESTRICT arow = ap + (qbase + i) * MemoryLen;
          score_row<ModelDim>(qi, kb, limit, hd, inv_scale, arow);
          softmax_inplace(arow, limit);
          for (std::size_t m = limit; m < MemoryLen; ++m) arow[m] = T{};
          // Context over memory rows, each read contiguously.
          T* ZF_RESTRICT ci = cp + (qbase + i) * ModelDim + off;
          for (std::size_t m = 0; m < limit; ++m) {
            const T weight = arow[m];
            const T* ZF_RESTRICT vm = vb + m * ModelDim;
            for (std::size_t d = 0; d < hd; ++d) ci[d] += weight * vm[d];
          }
        }
        for (std::size_t i = valid; i < QueryLen; ++i) {
          T* ZF_RESTRICT arow = ap + (qbase + i) * MemoryLen;
          for (std::size_t m = 0; m < MemoryLen; ++m) arow[m] = T{};
        }
      }
    }
    QuerySequence out;
    wo_.forward_spans(ctx_, out, query_spans_);
    return out;
  }

  // Returns dL/d(query input) and adds dL/d(memory) into *d_memory, which lets
  // a caller accumulate the encoder gradient across several decoder blocks that
  // all read the same memory. Must follow a matching forward().
  QuerySequence backward(const QuerySequence& d_out,
                         MemorySequence* d_memory) {
    const std::size_t hd = head_dim();
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    QuerySequence d_ctx;
    wo_.backward_spans(ctx_, d_out, d_ctx, query_spans_);

    QuerySequence dq;
    MemorySequence dk;
    MemorySequence dv;
    const T* ZF_RESTRICT qp = q_.data();
    const T* ZF_RESTRICT kp = k_.data();
    const T* ZF_RESTRICT vp = v_.data();
    const T* ZF_RESTRICT gp = d_ctx.data();
    T* ZF_RESTRICT dqp = dq.data();
    T* ZF_RESTRICT dkp = dk.data();
    T* ZF_RESTRICT dvp = dv.data();
    Tensor<T, MemoryLen> d_attn;
    T* ZF_RESTRICT dap = d_attn.data();

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      const T* ZF_RESTRICT ap = attn_[h].data();
      for (std::size_t b = 0; b < Batch; ++b) {
        const std::size_t valid = query_spans_[b].count;
        const std::size_t limit = memory_spans_[b].count;
        const std::size_t mbase = b * MemoryLen * ModelDim + off;
        for (std::size_t i = 0; i < valid; ++i) {
          const std::size_t row = b * QueryLen + i;
          attend_backward<ModelDim>(ap + row * MemoryLen, gp + row * ModelDim + off,
                                    qp + row * ModelDim + off, kp + mbase, vp + mbase,
                                    limit, hd, inv_scale, dap,
                                    dqp + row * ModelDim + off, dkp + mbase, dvp + mbase);
        }
      }
    }

    if (d_memory != nullptr) {
      wk_.backward_spans(memory_, dk, *d_memory, memory_spans_);
      wv_.backward_spans(memory_, dv, *d_memory, memory_spans_);
    }
    QuerySequence dx;
    wq_.backward_spans(q_in_, dq, dx, query_spans_);
    return dx;
  }

  void zero_grad() {
    wq_.zero_grad();
    wk_.zero_grad();
    wv_.zero_grad();
    wo_.zero_grad();
  }

  void apply_sgd(T lr) {
    wq_.apply_sgd(lr);
    wk_.apply_sgd(lr);
    wv_.apply_sgd(lr);
    wo_.apply_sgd(lr);
  }

  Linear<T, ModelDim, ModelDim>& query_proj() { return wq_; }
  Linear<T, ModelDim, ModelDim>& key_proj() { return wk_; }
  Linear<T, ModelDim, ModelDim>& value_proj() { return wv_; }
  Linear<T, ModelDim, ModelDim>& output_proj() { return wo_; }

  const Scores& attention_weights(std::size_t head) const {
    return attn_.at(head);
  }

 private:
  std::size_t heads_;
  std::array<RowSpan, Batch> query_spans_{};
  std::array<RowSpan, Batch> memory_spans_{};

  Linear<T, ModelDim, ModelDim> wq_;
  Linear<T, ModelDim, ModelDim> wk_;
  Linear<T, ModelDim, ModelDim> wv_;
  Linear<T, ModelDim, ModelDim> wo_;

  // activations cached by forward() for use by backward()
  QuerySequence q_in_;
  MemorySequence memory_;
  QuerySequence q_;
  MemorySequence k_;
  MemorySequence v_;
  QuerySequence ctx_;
  std::vector<Scores> attn_;
};

// One decoder block: self attention over the queries, then cross attention
// into the encoder memory, then a feed-forward. Pre-norm with residuals, the
// same arrangement TransformerBlock uses, with the cross-attention stage
// inserted between the two.
template <typename T, std::size_t QueryLen, std::size_t MemoryLen,
          std::size_t ModelDim, std::size_t FfHidden,
          Activation Act = Activation::kGelu, std::size_t Batch = 1>
class DecoderBlock {
 public:
  using value_type = T;
  using Cross = MultiHeadCrossAttention<T, QueryLen, MemoryLen, ModelDim, Batch>;
  using QuerySequence = typename Cross::QuerySequence;
  using MemorySequence = typename Cross::MemorySequence;

  static constexpr std::size_t parameter_count =
      MultiHeadSelfAttention<T, QueryLen, ModelDim>::parameter_count +
      Cross::parameter_count +
      FeedForward<T, QueryLen, ModelDim, FfHidden>::parameter_count +
      3 * LayerNorm<T, ModelDim>::parameter_count;

  using FF = FeedForward<T, Batch * QueryLen, ModelDim, FfHidden, Act>;

  explicit DecoderBlock(std::size_t heads)
      : self_(heads, /*causal=*/false), cross_(heads) {}

  std::size_t heads() const { return self_.heads(); }
  std::size_t rows() const { return self_.valid_length(); }
  RowSpans row_spans() const { return self_.row_spans(); }

  void set_valid_length(std::size_t length) {
    self_.set_valid_length(length);
    cross_.set_valid_length(length);
  }
  void set_valid_length(std::size_t b, std::size_t length) {
    self_.set_valid_length(b, length);
    cross_.set_valid_length(b, length);
  }
  void set_memory_length(std::size_t length) { cross_.set_memory_length(length); }
  void set_memory_length(std::size_t b, std::size_t length) {
    cross_.set_memory_length(b, length);
  }

  template <typename Rng>
  void randomize(Rng& rng) {
    self_.randomize(rng);
    cross_.randomize(rng);
    ff_.randomize(rng);
  }

  QuerySequence forward(const QuerySequence& x, const MemorySequence& memory) {
    const RowSpans rows = row_spans();
    x_ = x;
    ln1_out_ = ln1_.forward(x, rows);
    h1_ = add(x, self_.forward(ln1_out_));
    ln2_out_ = ln2_.forward(h1_, rows);
    h2_ = add(h1_, cross_.forward(ln2_out_, memory));
    ln3_out_ = ln3_.forward(h2_, rows);
    return add(h2_, ff_.forward(ln3_out_, rows));
  }

  QuerySequence backward(const QuerySequence& d_out,
                         MemorySequence* d_memory) {
    const RowSpans rows = row_spans();
    QuerySequence dh2 = d_out;
    add_into(dh2, ln3_.backward(h2_, ff_.backward(d_out), rows));

    QuerySequence dh1 = dh2;
    add_into(dh1, ln2_.backward(h1_, cross_.backward(dh2, d_memory), rows));

    QuerySequence dx = dh1;
    add_into(dx, ln1_.backward(x_, self_.backward(dh1), rows));
    return dx;
  }

  void zero_grad() {
    self_.zero_grad();
    cross_.zero_grad();
    ff_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ln3_.zero_grad();
  }

  void apply_sgd(T lr) {
    self_.apply_sgd(lr);
    cross_.apply_sgd(lr);
    ff_.apply_sgd(lr);
    ln1_.apply_sgd(lr);
    ln2_.apply_sgd(lr);
    ln3_.apply_sgd(lr);
  }

  MultiHeadSelfAttention<T, QueryLen, ModelDim, Batch>& self_attention() {
    return self_;
  }
  Cross& cross_attention() { return cross_; }
  FF& feed_forward() { return ff_; }
  LayerNorm<T, ModelDim>& self_norm() { return ln1_; }
  LayerNorm<T, ModelDim>& cross_norm() { return ln2_; }
  LayerNorm<T, ModelDim>& feed_forward_norm() { return ln3_; }

 private:
  MultiHeadSelfAttention<T, QueryLen, ModelDim, Batch> self_;
  Cross cross_;
  FF ff_;
  LayerNorm<T, ModelDim> ln1_;
  LayerNorm<T, ModelDim> ln2_;
  LayerNorm<T, ModelDim> ln3_;

  QuerySequence x_;
  QuerySequence ln1_out_;
  QuerySequence h1_;
  QuerySequence ln2_out_;
  QuerySequence h2_;
  QuerySequence ln3_out_;
};

// A stack of decoder blocks plus a final layer norm, all reading the same
// encoder memory. Depth is runtime, as in Transformer: every block has the
// same parameter shapes.
template <typename T, std::size_t QueryLen, std::size_t MemoryLen,
          std::size_t ModelDim, std::size_t FfHidden,
          Activation Act = Activation::kGelu, std::size_t Batch = 1>
class TransformerDecoder {
 public:
  using value_type = T;
  using Block = DecoderBlock<T, QueryLen, MemoryLen, ModelDim, FfHidden, Act, Batch>;
  using QuerySequence = typename Block::QuerySequence;
  using MemorySequence = typename Block::MemorySequence;

  static constexpr std::size_t query_len = QueryLen;
  static constexpr std::size_t memory_len = MemoryLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t batch = Batch;

  TransformerDecoder(std::size_t levels, std::size_t heads) {
    if (levels == 0) {
      throw std::invalid_argument("a decoder needs at least one level");
    }
    blocks_.reserve(levels);
    for (std::size_t i = 0; i < levels; ++i) {
      blocks_.emplace_back(heads);
    }
  }

  std::size_t depth() const { return blocks_.size(); }
  std::size_t heads() const { return blocks_.front().heads(); }
  Block& block(std::size_t i) { return blocks_.at(i); }
  const Block& block(std::size_t i) const { return blocks_.at(i); }

  void set_valid_length(std::size_t length) {
    for (Block& block : blocks_) block.set_valid_length(length);
  }
  void set_valid_length(std::size_t b, std::size_t length) {
    for (Block& block : blocks_) block.set_valid_length(b, length);
  }
  void set_memory_length(std::size_t length) {
    for (Block& block : blocks_) block.set_memory_length(length);
  }
  void set_memory_length(std::size_t b, std::size_t length) {
    for (Block& block : blocks_) block.set_memory_length(b, length);
  }

  template <typename Rng>
  void randomize(Rng& rng) {
    for (Block& block : blocks_) {
      block.randomize(rng);
    }
  }

  QuerySequence forward(const QuerySequence& x, const MemorySequence& memory) {
    QuerySequence h = x;
    for (Block& block : blocks_) {
      h = block.forward(h, memory);
      ZF_CHECK_FINITE(h, "decoder block forward");
    }
    out_pre_norm_ = h;
    return out_norm_.forward(h, blocks_.front().row_spans());
  }

  // Returns dL/dx and accumulates dL/d(memory) into *d_memory. Every block
  // read the same memory, so every block's cross-attention contributes.
  QuerySequence backward(const QuerySequence& d_out,
                         MemorySequence* d_memory) {
    QuerySequence d =
        out_norm_.backward(out_pre_norm_, d_out, blocks_.front().row_spans());
    for (std::size_t i = blocks_.size(); i-- > 0;) {
      d = blocks_[i].backward(d, d_memory);
      ZF_CHECK_FINITE(d, "decoder block backward");
    }
    return d;
  }

  void zero_grad() {
    for (Block& block : blocks_) {
      block.zero_grad();
    }
    out_norm_.zero_grad();
  }

  void apply_sgd(T lr) {
    for (Block& block : blocks_) {
      block.apply_sgd(lr);
    }
    out_norm_.apply_sgd(lr);
  }

  LayerNorm<T, ModelDim>& output_norm() { return out_norm_; }

 private:
  std::vector<Block> blocks_;
  LayerNorm<T, ModelDim> out_norm_;
  QuerySequence out_pre_norm_;
};

}  // namespace zf
