#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "add.h"
#include "attention.h"
#include "layers.h"
#include "tensor.h"

namespace zf {

// Position-wise feed-forward sublayer: ModelDim -> FfHidden -> ModelDim,
// applied to every row of the sequence with shared weights.
template <typename T, std::size_t SeqLen, std::size_t ModelDim,
          std::size_t FfHidden>
class FeedForward {
 public:
  using Sequence = Tensor<T, SeqLen, ModelDim>;
  using HiddenSequence = Tensor<T, SeqLen, FfHidden>;

  static constexpr std::size_t parameter_count =
      Linear<T, ModelDim, FfHidden>::parameter_count +
      Linear<T, FfHidden, ModelDim>::parameter_count;

  explicit FeedForward(Activation act = Activation::kGelu) : act_(act) {}

  Activation activation() const { return act_; }
  void set_activation(Activation act) { act_ = act; }

  template <typename Rng>
  void randomize(Rng& rng) {
    fc1_.randomize(rng);
    fc2_.randomize(rng);
  }

  // `rows` is how many rows are real; see Linear's note on padded sequences.
  Sequence forward(const Sequence& x, std::size_t rows = SeqLen) {
    const RowSpan all{0, rows < SeqLen ? rows : SeqLen};
    return forward(x, RowSpans(&all, 1));
  }

  // Only the rows inside `spans` -- e.g. the real rows of each example in a
  // batch. The rest of the result is zero.
  Sequence forward(const Sequence& x, RowSpans spans) {
    x_ = x;
    spans_.assign(spans.begin(), spans.end());
    h_pre_.fill(T{});
    fc1_.forward_spans(x, h_pre_, spans);
    for (const RowSpan& s : spans_) {
      // The derivative is stored now, while its tanh is at hand.
      for (std::size_t i = s.start * FfHidden; i < (s.start + s.count) * FfHidden; ++i) {
        h_[i] = activation_apply_with_grad(act_, h_pre_[i], &dact_[i]);
      }
    }
    Sequence out;
    fc2_.forward_spans(h_, out, spans);
    return out;
  }

  Sequence backward(const Sequence& d_out) {
    HiddenSequence dh;
    fc2_.backward_spans(h_, d_out, dh, spans_);
    for (const RowSpan& s : spans_) {
      for (std::size_t i = s.start * FfHidden; i < (s.start + s.count) * FfHidden; ++i) {
        dh[i] *= dact_[i];
      }
    }
    Sequence dx;
    fc1_.backward_spans(x_, dh, dx, spans_);
    return dx;
  }

  void zero_grad() {
    fc1_.zero_grad();
    fc2_.zero_grad();
  }

  void apply_sgd(T lr) {
    fc1_.apply_sgd(lr);
    fc2_.apply_sgd(lr);
  }

  Linear<T, ModelDim, FfHidden>& input_layer() { return fc1_; }
  Linear<T, FfHidden, ModelDim>& output_layer() { return fc2_; }

 private:
  Activation act_;
  Linear<T, ModelDim, FfHidden> fc1_;
  Linear<T, FfHidden, ModelDim> fc2_;

  std::vector<RowSpan> spans_;
  Sequence x_;
  HiddenSequence h_pre_;
  HiddenSequence h_;
  HiddenSequence dact_;  // activation derivative at h_pre_
};

// One pre-norm transformer block:
//   h = x + attention(layernorm(x))
//   y = h + feedforward(layernorm(h))
//
// Sequence length, model width and FFN width are template parameters, so all
// parameter tensors are shape-checked. Head count, causal masking and the FFN
// activation are runtime settings; none of them changes a parameter shape.
// `Batch` stacks that many independent sequences row-wise; see
// MultiHeadSelfAttention.
template <typename T, std::size_t SeqLen, std::size_t ModelDim,
          std::size_t FfHidden, std::size_t Batch = 1>
class TransformerBlock {
  static_assert(std::is_floating_point_v<T>,
                "trainable models require a floating point element type");

 public:
  using value_type = T;
  using Attention = MultiHeadSelfAttention<T, SeqLen, ModelDim, Batch>;
  using Sequence = typename Attention::Sequence;

  static constexpr std::size_t seq_len = SeqLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t ff_hidden = FfHidden;
  static constexpr std::size_t batch = Batch;
  static constexpr std::size_t parameter_count =
      Attention::parameter_count +
      FeedForward<T, SeqLen, ModelDim, FfHidden>::parameter_count +
      2 * LayerNorm<T, ModelDim>::parameter_count;

  explicit TransformerBlock(std::size_t heads, bool causal = false,
                            Activation act = Activation::kGelu)
      : attn_(heads, causal), ff_(act) {}

  std::size_t heads() const { return attn_.heads(); }
  bool causal() const { return attn_.causal(); }
  void set_causal(bool causal) { attn_.set_causal(causal); }
  std::size_t valid_length() const { return attn_.valid_length(); }
  std::size_t valid_length(std::size_t b) const { return attn_.valid_length(b); }
  void set_valid_length(std::size_t length) { attn_.set_valid_length(length); }
  void set_valid_length(std::size_t b, std::size_t length) {
    attn_.set_valid_length(b, length);
  }
  RowSpans row_spans() const { return attn_.row_spans(); }

  template <typename Rng>
  void randomize(Rng& rng) {
    attn_.randomize(rng);
    ff_.randomize(rng);
  }

  Sequence forward(const Sequence& x) {
    // Which rows are real. Every sublayer gets the same answer, so a padded
    // tail is skipped by the whole block rather than only by the attention.
    const RowSpans rows = row_spans();
    x_ = x;
    ln1_out_ = ln1_.forward(x, rows);
    h_ = add(x, attn_.forward(ln1_out_));
    ln2_out_ = ln2_.forward(h_, rows);
    return add(h_, ff_.forward(ln2_out_, rows));
  }

  // Returns dL/dx. Must follow a matching forward().
  Sequence backward(const Sequence& d_out) {
    const RowSpans rows = row_spans();
    // y = h + ff(ln2(h)): the residual sends d_out down both paths
    Sequence dh = d_out;
    add_into(dh, ln2_.backward(h_, ff_.backward(d_out), rows));

    // h = x + attn(ln1(x)): likewise
    Sequence dx = dh;
    add_into(dx, ln1_.backward(x_, attn_.backward(dh), rows));
    return dx;
  }

  void zero_grad() {
    attn_.zero_grad();
    ff_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
  }

  void apply_sgd(T lr) {
    attn_.apply_sgd(lr);
    ff_.apply_sgd(lr);
    ln1_.apply_sgd(lr);
    ln2_.apply_sgd(lr);
  }

  Attention& attention() { return attn_; }
  FeedForward<T, Batch * SeqLen, ModelDim, FfHidden>& feed_forward() { return ff_; }
  LayerNorm<T, ModelDim>& attention_norm() { return ln1_; }
  LayerNorm<T, ModelDim>& feed_forward_norm() { return ln2_; }

 private:
  Attention attn_;
  FeedForward<T, Batch * SeqLen, ModelDim, FfHidden> ff_;
  LayerNorm<T, ModelDim> ln1_;
  LayerNorm<T, ModelDim> ln2_;

  // activations cached by forward() for use by backward()
  Sequence x_;
  Sequence ln1_out_;
  Sequence h_;
  Sequence ln2_out_;
};

// A stack of identical blocks plus a final layer norm. Depth is runtime: every
// block has the same parameter shapes, so it cannot affect type safety.
template <typename T, std::size_t SeqLen, std::size_t ModelDim,
          std::size_t FfHidden, std::size_t Batch = 1>
class Transformer {
  static_assert(std::is_floating_point_v<T>,
                "trainable models require a floating point element type");

 public:
  using value_type = T;
  using Block = TransformerBlock<T, SeqLen, ModelDim, FfHidden, Batch>;
  using Sequence = typename Block::Sequence;

  static constexpr std::size_t seq_len = SeqLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t ff_hidden = FfHidden;

  Transformer(std::size_t depth, std::size_t heads, bool causal = false,
              Activation act = Activation::kGelu) {
    if (depth == 0) {
      throw std::invalid_argument("transformer needs at least one block");
    }
    blocks_.reserve(depth);
    for (std::size_t i = 0; i < depth; ++i) {
      blocks_.emplace_back(heads, causal, act);
    }
  }

  std::size_t depth() const { return blocks_.size(); }
  std::size_t heads() const { return blocks_.front().heads(); }

  std::size_t valid_length() const { return blocks_.front().valid_length(); }

  // Masks trailing key positions in every block, so a padded sequence can
  // share one compile-time SeqLen.
  void set_valid_length(std::size_t length) {
    for (Block& block : blocks_) {
      block.set_valid_length(length);
    }
  }
  void set_valid_length(std::size_t b, std::size_t length) {
    for (Block& block : blocks_) {
      block.set_valid_length(b, length);
    }
  }

  std::size_t parameter_count() const {
    return blocks_.size() * Block::parameter_count +
           LayerNorm<T, ModelDim>::parameter_count;
  }

  template <typename Rng>
  void randomize(Rng& rng) {
    for (Block& b : blocks_) {
      b.randomize(rng);
    }
  }

  Sequence forward(const Sequence& x) {
    Sequence cur = x;
    for (Block& b : blocks_) {
      cur = b.forward(cur);
      ZF_CHECK_FINITE(cur, "transformer block forward");
    }
    final_in_ = cur;
    return ln_f_.forward(cur, blocks_.front().row_spans());
  }

  // Returns dL/dx. Must follow a matching forward().
  Sequence backward(const Sequence& d_out) {
    Sequence d = ln_f_.backward(final_in_, d_out, blocks_.front().row_spans());
    for (std::size_t i = blocks_.size(); i-- > 0;) {
      d = blocks_[i].backward(d);
      ZF_CHECK_FINITE(d, "transformer block backward");
    }
    return d;
  }

  void zero_grad() {
    for (Block& b : blocks_) {
      b.zero_grad();
    }
    ln_f_.zero_grad();
  }

  void apply_sgd(T lr) {
    for (Block& b : blocks_) {
      b.apply_sgd(lr);
    }
    ln_f_.apply_sgd(lr);
  }

  Block& block(std::size_t i) { return blocks_.at(i); }
  LayerNorm<T, ModelDim>& final_norm() { return ln_f_; }

 private:
  std::vector<Block> blocks_;
  LayerNorm<T, ModelDim> ln_f_;

  Sequence final_in_;
};

}  // namespace zf
