#pragma once

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
template <typename T, std::size_t QueryLen, std::size_t MemoryLen,
          std::size_t ModelDim>
class MultiHeadCrossAttention {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  using QuerySequence = Tensor<T, QueryLen, ModelDim>;
  using MemorySequence = Tensor<T, MemoryLen, ModelDim>;
  using Scores = Tensor<T, QueryLen, MemoryLen>;

  static constexpr std::size_t query_len = QueryLen;
  static constexpr std::size_t memory_len = MemoryLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t parameter_count =
      4 * Linear<T, ModelDim, ModelDim>::parameter_count;

  explicit MultiHeadCrossAttention(std::size_t heads) : heads_(heads) {
    if (heads == 0 || ModelDim % heads != 0) {
      throw std::invalid_argument(
          "head count " + std::to_string(heads) +
          " must be non-zero and divide model dim " + std::to_string(ModelDim));
    }
    attn_.assign(heads_, Scores{});
  }

  std::size_t heads() const { return heads_; }
  std::size_t head_dim() const { return ModelDim / heads_; }

  std::size_t memory_length() const { return memory_length_; }
  // Throws std::out_of_range unless 1 <= length <= MemoryLen. Zero would leave
  // every query with nothing to attend to and no softmax to take.
  void set_memory_length(std::size_t length) {
    if (length == 0 || length > MemoryLen) {
      throw std::out_of_range("memory length " + std::to_string(length) +
                              " outside [1, " + std::to_string(MemoryLen) +
                              "]");
    }
    memory_length_ = length;
  }

  std::size_t valid_length() const { return valid_length_; }
  void set_valid_length(std::size_t length) {
    if (length == 0 || length > QueryLen) {
      throw std::out_of_range("valid length " + std::to_string(length) +
                              " outside [1, " + std::to_string(QueryLen) + "]");
    }
    valid_length_ = length;
  }

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
    q_ = wq_.template forward<QueryLen>(q_in, valid_length_);
    k_ = wk_.template forward<MemoryLen>(memory, memory_length_);
    v_ = wv_.template forward<MemoryLen>(memory, memory_length_);
    ctx_ = QuerySequence{};

    const std::size_t limit = memory_length_;
    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      Scores& a = attn_[h];
      a.fill(T{});
      for (std::size_t i = 0; i < valid_length_; ++i) {
        for (std::size_t j = 0; j < limit; ++j) {
          T s{};
          for (std::size_t d = 0; d < hd; ++d) {
            s += q_(i, off + d) * k_(j, off + d);
          }
          a(i, j) = s * inv_scale;
        }
        softmax_inplace(&a(i, 0), limit);
        for (std::size_t j = limit; j < MemoryLen; ++j) {
          a(i, j) = T{};
        }
        for (std::size_t d = 0; d < hd; ++d) {
          T acc{};
          for (std::size_t j = 0; j < limit; ++j) {
            acc += a(i, j) * v_(j, off + d);
          }
          ctx_(i, off + d) = acc;
        }
      }
    }
    return wo_.template forward<QueryLen>(ctx_, valid_length_);
  }

  // Returns dL/d(query input) and adds dL/d(memory) into *d_memory, which lets
  // a caller accumulate the encoder gradient across several decoder blocks that
  // all read the same memory. Must follow a matching forward().
  QuerySequence backward(const QuerySequence& d_out,
                         MemorySequence* d_memory) {
    const std::size_t hd = head_dim();
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    const QuerySequence d_ctx =
        wo_.template backward<QueryLen>(ctx_, d_out, valid_length_);

    QuerySequence dq;
    MemorySequence dk;
    MemorySequence dv;
    const std::size_t limit = memory_length_;
    // One row of scratch, hoisted: a fresh allocation per head per query row
    // is a lot of allocating to do nothing with.
    Tensor<T, MemoryLen> d_attn;

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      const Scores& a = attn_[h];
      for (std::size_t i = 0; i < valid_length_; ++i) {
        for (std::size_t j = 0; j < limit; ++j) {
          T acc{};
          for (std::size_t d = 0; d < hd; ++d) {
            acc += d_ctx(i, off + d) * v_(j, off + d);
          }
          d_attn(j) = acc;
          for (std::size_t d = 0; d < hd; ++d) {
            dv(j, off + d) += a(i, j) * d_ctx(i, off + d);
          }
        }

        // softmax jacobian: ds_j = p_j * (dp_j - sum_m p_m dp_m)
        T dot_pd{};
        for (std::size_t j = 0; j < limit; ++j) {
          dot_pd += a(i, j) * d_attn(j);
        }
        for (std::size_t j = 0; j < limit; ++j) {
          const T ds = a(i, j) * (d_attn(j) - dot_pd) * inv_scale;
          for (std::size_t d = 0; d < hd; ++d) {
            dq(i, off + d) += ds * k_(j, off + d);
            dk(j, off + d) += ds * q_(i, off + d);
          }
        }
      }
    }

    if (d_memory != nullptr) {
      add_into(*d_memory,
               wk_.template backward<MemoryLen>(memory_, dk, memory_length_));
      add_into(*d_memory,
               wv_.template backward<MemoryLen>(memory_, dv, memory_length_));
    }
    return wq_.template backward<QueryLen>(q_in_, dq, valid_length_);
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
  std::size_t memory_length_ = MemoryLen;
  std::size_t valid_length_ = QueryLen;

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
          std::size_t ModelDim, std::size_t FfHidden>
class DecoderBlock {
 public:
  using value_type = T;
  using QuerySequence = Tensor<T, QueryLen, ModelDim>;
  using MemorySequence = Tensor<T, MemoryLen, ModelDim>;

  static constexpr std::size_t parameter_count =
      MultiHeadSelfAttention<T, QueryLen, ModelDim>::parameter_count +
      MultiHeadCrossAttention<T, QueryLen, MemoryLen,
                              ModelDim>::parameter_count +
      FeedForward<T, QueryLen, ModelDim, FfHidden>::parameter_count +
      3 * LayerNorm<T, ModelDim>::parameter_count;

  explicit DecoderBlock(std::size_t heads, Activation act = Activation::kGelu)
      : self_(heads, /*causal=*/false), cross_(heads), ff_(act) {}

  std::size_t heads() const { return self_.heads(); }
  // How many decoder positions are real.
  std::size_t rows() const { return self_.valid_length(); }

  // Masks the padded tail of both sides: `length` decoder positions are real,
  // `memory` encoder positions are real.
  void set_valid_length(std::size_t length) {
    self_.set_valid_length(length);
    cross_.set_valid_length(length);
  }
  void set_memory_length(std::size_t length) {
    cross_.set_memory_length(length);
  }

  template <typename Rng>
  void randomize(Rng& rng) {
    self_.randomize(rng);
    cross_.randomize(rng);
    ff_.randomize(rng);
  }

  QuerySequence forward(const QuerySequence& x, const MemorySequence& memory) {
    // One row count for the whole block, as in TransformerBlock.
    const std::size_t rows = self_.valid_length();
    x_ = x;
    ln1_out_ = ln1_.template forward<QueryLen>(x, rows);
    h1_ = add(x, self_.forward(ln1_out_));
    ln2_out_ = ln2_.template forward<QueryLen>(h1_, rows);
    h2_ = add(h1_, cross_.forward(ln2_out_, memory));
    ln3_out_ = ln3_.template forward<QueryLen>(h2_, rows);
    return add(h2_, ff_.forward(ln3_out_, rows));
  }

  // Returns dL/dx and accumulates dL/d(memory) into *d_memory.
  QuerySequence backward(const QuerySequence& d_out,
                         MemorySequence* d_memory) {
    const std::size_t rows = self_.valid_length();
    QuerySequence dh2 = d_out;
    add_into(dh2,
             ln3_.template backward<QueryLen>(h2_, ff_.backward(d_out), rows));

    QuerySequence dh1 = dh2;
    add_into(dh1, ln2_.template backward<QueryLen>(
                      h1_, cross_.backward(dh2, d_memory), rows));

    QuerySequence dx = dh1;
    add_into(dx,
             ln1_.template backward<QueryLen>(x_, self_.backward(dh1), rows));
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

  MultiHeadSelfAttention<T, QueryLen, ModelDim>& self_attention() {
    return self_;
  }
  MultiHeadCrossAttention<T, QueryLen, MemoryLen, ModelDim>& cross_attention() {
    return cross_;
  }
  FeedForward<T, QueryLen, ModelDim, FfHidden>& feed_forward() { return ff_; }
  LayerNorm<T, ModelDim>& self_norm() { return ln1_; }
  LayerNorm<T, ModelDim>& cross_norm() { return ln2_; }
  LayerNorm<T, ModelDim>& feed_forward_norm() { return ln3_; }

 private:
  MultiHeadSelfAttention<T, QueryLen, ModelDim> self_;
  MultiHeadCrossAttention<T, QueryLen, MemoryLen, ModelDim> cross_;
  FeedForward<T, QueryLen, ModelDim, FfHidden> ff_;
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
          std::size_t ModelDim, std::size_t FfHidden>
class TransformerDecoder {
 public:
  using value_type = T;
  using QuerySequence = Tensor<T, QueryLen, ModelDim>;
  using MemorySequence = Tensor<T, MemoryLen, ModelDim>;
  using Block = DecoderBlock<T, QueryLen, MemoryLen, ModelDim, FfHidden>;

  static constexpr std::size_t query_len = QueryLen;
  static constexpr std::size_t memory_len = MemoryLen;
  static constexpr std::size_t model_dim = ModelDim;

  TransformerDecoder(std::size_t levels, std::size_t heads,
                     Activation act = Activation::kGelu) {
    if (levels == 0) {
      throw std::invalid_argument("a decoder needs at least one level");
    }
    blocks_.reserve(levels);
    for (std::size_t i = 0; i < levels; ++i) {
      blocks_.emplace_back(heads, act);
    }
  }

  std::size_t depth() const { return blocks_.size(); }
  std::size_t heads() const { return blocks_.front().heads(); }
  Block& block(std::size_t i) { return blocks_.at(i); }
  const Block& block(std::size_t i) const { return blocks_.at(i); }

  void set_valid_length(std::size_t length) {
    for (Block& block : blocks_) {
      block.set_valid_length(length);
    }
  }
  void set_memory_length(std::size_t length) {
    for (Block& block : blocks_) {
      block.set_memory_length(length);
    }
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
    }
    out_pre_norm_ = h;
    return out_norm_.template forward<QueryLen>(h, blocks_.front().rows());
  }

  // Returns dL/dx and accumulates dL/d(memory) into *d_memory. Every block
  // read the same memory, so every block's cross-attention contributes.
  QuerySequence backward(const QuerySequence& d_out,
                         MemorySequence* d_memory) {
    QuerySequence d = out_norm_.template backward<QueryLen>(
        out_pre_norm_, d_out, blocks_.front().rows());
    for (std::size_t i = blocks_.size(); i-- > 0;) {
      d = blocks_[i].backward(d, d_memory);
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
