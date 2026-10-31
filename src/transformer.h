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
    x_ = x;
    rows_ = rows < SeqLen ? rows : SeqLen;
    h_pre_ = fc1_.template forward<SeqLen>(x, rows_);
    const std::size_t active = rows_ * FfHidden;
    for (std::size_t i = 0; i < active; ++i) {
      h_[i] = activation_apply(act_, h_pre_[i]);
    }
    return fc2_.template forward<SeqLen>(h_, rows_);
  }

  Sequence backward(const Sequence& d_out) {
    HiddenSequence dh = fc2_.template backward<SeqLen>(h_, d_out, rows_);
    const std::size_t active = rows_ * FfHidden;
    for (std::size_t i = 0; i < active; ++i) {
      dh[i] *= activation_grad(act_, h_pre_[i]);
    }
    return fc1_.template backward<SeqLen>(x_, dh, rows_);
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

  std::size_t rows_ = SeqLen;
  Sequence x_;
  HiddenSequence h_pre_;
  HiddenSequence h_;
};

// One pre-norm transformer block:
//   h = x + attention(layernorm(x))
//   y = h + feedforward(layernorm(h))
//
// Sequence length, model width and FFN width are template parameters, so all
// parameter tensors are shape-checked. Head count, causal masking and the FFN
// activation are runtime settings; none of them changes a parameter shape.
template <typename T, std::size_t SeqLen, std::size_t ModelDim,
          std::size_t FfHidden>
class TransformerBlock {
  static_assert(std::is_floating_point_v<T>,
                "trainable models require a floating point element type");

 public:
  using value_type = T;
  using Sequence = Tensor<T, SeqLen, ModelDim>;

  static constexpr std::size_t seq_len = SeqLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t ff_hidden = FfHidden;
  static constexpr std::size_t parameter_count =
      MultiHeadSelfAttention<T, SeqLen, ModelDim>::parameter_count +
      FeedForward<T, SeqLen, ModelDim, FfHidden>::parameter_count +
      2 * LayerNorm<T, ModelDim>::parameter_count;

  explicit TransformerBlock(std::size_t heads, bool causal = false,
                            Activation act = Activation::kGelu)
      : attn_(heads, causal), ff_(act) {}

  std::size_t heads() const { return attn_.heads(); }
  bool causal() const { return attn_.causal(); }
  void set_causal(bool causal) { attn_.set_causal(causal); }
  std::size_t valid_length() const { return attn_.valid_length(); }
  void set_valid_length(std::size_t length) {
    attn_.set_valid_length(length);
  }

  template <typename Rng>
  void randomize(Rng& rng) {
    attn_.randomize(rng);
    ff_.randomize(rng);
  }

  Sequence forward(const Sequence& x) {
    // How much of the sequence is real. Every sublayer gets the same answer,
    // so a padded tail is skipped by the whole block rather than only by the
    // attention inside it.
    const std::size_t rows = attn_.valid_length();
    x_ = x;
    ln1_out_ = ln1_.template forward<SeqLen>(x, rows);
    h_ = add(x, attn_.forward(ln1_out_));
    ln2_out_ = ln2_.template forward<SeqLen>(h_, rows);
    return add(h_, ff_.forward(ln2_out_, rows));
  }

  // Returns dL/dx. Must follow a matching forward().
  Sequence backward(const Sequence& d_out) {
    const std::size_t rows = attn_.valid_length();
    // y = h + ff(ln2(h)): the residual sends d_out down both paths
    Sequence dh = d_out;
    add_into(dh,
             ln2_.template backward<SeqLen>(h_, ff_.backward(d_out), rows));

    // h = x + attn(ln1(x)): likewise
    Sequence dx = dh;
    add_into(dx,
             ln1_.template backward<SeqLen>(x_, attn_.backward(dh), rows));
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

  MultiHeadSelfAttention<T, SeqLen, ModelDim>& attention() { return attn_; }
  FeedForward<T, SeqLen, ModelDim, FfHidden>& feed_forward() { return ff_; }
  LayerNorm<T, ModelDim>& attention_norm() { return ln1_; }
  LayerNorm<T, ModelDim>& feed_forward_norm() { return ln2_; }

 private:
  MultiHeadSelfAttention<T, SeqLen, ModelDim> attn_;
  FeedForward<T, SeqLen, ModelDim, FfHidden> ff_;
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
          std::size_t FfHidden>
class Transformer {
  static_assert(std::is_floating_point_v<T>,
                "trainable models require a floating point element type");

 public:
  using value_type = T;
  using Sequence = Tensor<T, SeqLen, ModelDim>;
  using Block = TransformerBlock<T, SeqLen, ModelDim, FfHidden>;

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
    }
    final_in_ = cur;
    return ln_f_.template forward<SeqLen>(cur);
  }

  // Returns dL/dx. Must follow a matching forward().
  Sequence backward(const Sequence& d_out) {
    Sequence d =
        ln_f_.template backward<SeqLen>(final_in_, d_out, valid_length());
    for (std::size_t i = blocks_.size(); i-- > 0;) {
      d = blocks_[i].backward(d);
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
