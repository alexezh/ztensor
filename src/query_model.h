#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dot.h"
#include "layers.h"
#include "mlp.h"
#include "tensor.h"

namespace zf {

// Retrieval-style query encoder (one tower of a two-tower model):
//
//   token ids -> embedding -> mean pool -> MLP projection -> L2 normalize
//
// Vocabulary, embedding width, projection width and output width are template
// parameters, so an embedding of the wrong width cannot be scored against this
// tower's output. The token count is runtime (queries vary in length) and so is
// the projection activation.
//
// Instantiating the same class a second time gives the document tower; both
// produce Tensor<T, OutDim>, and `score` is their dot product. Because the
// outputs are L2-normalized, that dot product is cosine similarity.
template <typename T, std::size_t VocabSize, std::size_t EmbedDim,
          std::size_t HiddenDim, std::size_t OutDim>
class QueryModel {
  static_assert(is_real_v<T>,
                "trainable models require a floating point element type");

 public:
  using value_type = T;
  using Acc = accumulator_t<T>;
  using Embedded = Tensor<T, EmbedDim>;
  using Output = Tensor<T, OutDim>;

  static constexpr std::size_t vocab_size = VocabSize;
  static constexpr std::size_t embed_dim = EmbedDim;
  static constexpr std::size_t hidden_dim = HiddenDim;
  static constexpr std::size_t out_dim = OutDim;
  static constexpr std::size_t parameter_count =
      Embedding<T, VocabSize, EmbedDim>::parameter_count +
      Mlp<T, EmbedDim, HiddenDim, OutDim>::parameter_count;

  explicit QueryModel(Activation act = Activation::kRelu,
                      bool normalize = true)
      : proj_(act), normalize_(normalize) {}

  bool normalize() const { return normalize_; }
  void set_normalize(bool normalize) { normalize_ = normalize; }
  Activation activation() const { return proj_.activation(); }

  template <typename Rng>
  void randomize(Rng& rng) {
    embed_.randomize(rng);
    proj_.randomize(rng);
  }

  // Encodes a bag of token ids. Throws std::invalid_argument when empty and
  // std::out_of_range when an id is outside the vocabulary.
  Output forward(const std::vector<std::size_t>& token_ids) {
    if (token_ids.empty()) {
      throw std::invalid_argument("query model needs at least one token");
    }
    tokens_ = token_ids;

    Acc pooled[EmbedDim] = {};
    for (std::size_t id : token_ids) {
      const Embedded e = embed_.forward(id);
      for (std::size_t i = 0; i < EmbedDim; ++i) {
        pooled[i] += e(i);
      }
    }
    const Acc inv_n = Acc{1} / static_cast<Acc>(token_ids.size());
    for (std::size_t i = 0; i < EmbedDim; ++i) {
      pooled_(i) = static_cast<T>(pooled[i] * inv_n);
    }

    pre_norm_ = proj_.forward(pooled_);
    if (!normalize_) {
      out_ = pre_norm_;
      return out_;
    }

    Acc sq{};
    for (std::size_t i = 0; i < OutDim; ++i) {
      sq += pre_norm_(i) * pre_norm_(i);
    }
    inv_norm_ = Acc{1} / std::sqrt(sq + eps_);
    out_ = Output{};
    for (std::size_t i = 0; i < OutDim; ++i) {
      out_(i) = static_cast<T>(pre_norm_(i) * inv_norm_);
    }
    return out_;
  }

  // Accumulates parameter gradients. There is no input gradient to return:
  // the input is a list of token ids, and the embedding rows it touched take
  // the gradient instead. Must follow a matching forward().
  void backward(const Output& d_out) {
    if (tokens_.empty()) {
      throw std::logic_error("backward() called before forward()");
    }

    Output d_pre = d_out;
    if (normalize_) {
      // y = x / ||x||  =>  dx = (dy - y (y . dy)) / ||x||
      Acc proj{};
      for (std::size_t i = 0; i < OutDim; ++i) {
        proj += out_(i) * d_out(i);
      }
      for (std::size_t i = 0; i < OutDim; ++i) {
        d_pre(i) = static_cast<T>((d_out(i) - out_(i) * proj) * inv_norm_);
      }
    }

    const Embedded d_pooled = proj_.backward(d_pre);

    // mean pooling spreads the gradient evenly over the tokens
    const Acc inv_n = Acc{1} / static_cast<Acc>(tokens_.size());
    Embedded d_token;
    for (std::size_t i = 0; i < EmbedDim; ++i) {
      d_token(i) = static_cast<T>(d_pooled(i) * inv_n);
    }
    for (std::size_t id : tokens_) {
      embed_.backward(id, d_token);
    }
  }

  // Cosine similarity when both towers normalize, plain dot product otherwise.
  static Acc score(const Output& a, const Output& b) { return dot(a, b); }

  void zero_grad() {
    embed_.zero_grad();
    proj_.zero_grad();
  }

  void apply_sgd(Acc lr) {
    embed_.apply_sgd(lr);
    proj_.apply_sgd(lr);
  }

  Embedding<T, VocabSize, EmbedDim>& embedding() { return embed_; }
  const Embedding<T, VocabSize, EmbedDim>& embedding() const { return embed_; }
  Mlp<T, EmbedDim, HiddenDim, OutDim>& projection() { return proj_; }
  const Mlp<T, EmbedDim, HiddenDim, OutDim>& projection() const {
    return proj_;
  }

  // Pooled embedding from the last forward(), before projection.
  const Embedded& pooled() const { return pooled_; }

 private:
  Embedding<T, VocabSize, EmbedDim> embed_;
  Mlp<T, EmbedDim, HiddenDim, OutDim> proj_;
  bool normalize_;
  Acc eps_ = static_cast<Acc>(1e-12);

  // activations cached by forward() for use by backward()
  std::vector<std::size_t> tokens_;
  Embedded pooled_;
  Output pre_norm_;
  Output out_;
  Acc inv_norm_ = Acc{1};
};

}  // namespace zf
