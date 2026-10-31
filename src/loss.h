#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dot.h"
#include "layers.h"
#include "tensor.h"

namespace zf {

// Losses follow the same two-step shape as the models: forward() returns the
// scalar loss and caches what it needs, backward() returns dL/d(prediction) to
// hand straight to a model's backward().

// Mean squared error over the prediction, averaged across all elements.
template <typename T, std::size_t... Dims>
class MseLoss {
  static_assert(std::is_floating_point_v<T>,
                "losses require a floating point element type");

 public:
  using value_type = T;
  using Prediction = Tensor<T, Dims...>;

  T forward(const Prediction& pred, const Prediction& target) {
    T sum{};
    for (std::size_t i = 0; i < pred.size(); ++i) {
      const T d = pred[i] - target[i];
      d_pred_[i] = static_cast<T>(2) * d / static_cast<T>(pred.size());
      sum += d * d;
    }
    ready_ = true;
    return sum / static_cast<T>(pred.size());
  }

  const Prediction& backward() const {
    if (!ready_) {
      throw std::logic_error("backward() called before forward()");
    }
    return d_pred_;
  }

 private:
  Prediction d_pred_;
  bool ready_ = false;
};

// Softmax over the logits followed by cross entropy against a class index.
// The softmax is fused into the loss, so the gradient is just p - onehot.
template <typename T, std::size_t NumClasses>
class SoftmaxCrossEntropyLoss {
  static_assert(std::is_floating_point_v<T>,
                "losses require a floating point element type");
  static_assert(NumClasses > 1, "cross entropy needs at least two classes");

 public:
  using value_type = T;
  using Logits = Tensor<T, NumClasses>;

  static constexpr std::size_t num_classes = NumClasses;

  // Throws std::out_of_range when `label` is not a valid class index.
  T forward(const Logits& logits, std::size_t label) {
    if (label >= NumClasses) {
      throw std::out_of_range("class label " + std::to_string(label) +
                              " outside [0, " + std::to_string(NumClasses) +
                              ")");
    }
    probs_ = logits;
    softmax_inplace(probs_.data(), NumClasses);

    for (std::size_t i = 0; i < NumClasses; ++i) {
      d_logits_(i) = probs_(i);
    }
    d_logits_(label) -= T{1};
    ready_ = true;

    const T p = probs_(label);
    const T floor = std::numeric_limits<T>::min();
    return -std::log(p > floor ? p : floor);
  }

  const Logits& backward() const {
    if (!ready_) {
      throw std::logic_error("backward() called before forward()");
    }
    return d_logits_;
  }

  // Softmax probabilities from the last forward().
  const Logits& probabilities() const { return probs_; }

  std::size_t argmax() const {
    std::size_t best = 0;
    for (std::size_t i = 1; i < NumClasses; ++i) {
      if (probs_(i) > probs_(best)) {
        best = i;
      }
    }
    return best;
  }

 private:
  Logits probs_;
  Logits d_logits_;
  bool ready_ = false;
};

// InfoNCE / sampled softmax over candidate embeddings — the loss that trains a
// two-tower retrieval model. Logits are dot(query, candidate) / temperature;
// the loss is the cross entropy of picking the positive candidate.
//
// The candidate count is runtime (the number of negatives you sample is a
// training knob, not a shape), while the embedding width is a template
// parameter, so a query tower cannot be scored against a differently sized
// document tower.
template <typename T, std::size_t EmbedDim>
class InfoNceLoss {
  static_assert(std::is_floating_point_v<T>,
                "losses require a floating point element type");

 public:
  using value_type = T;
  using Vector = Tensor<T, EmbedDim>;

  static constexpr std::size_t embed_dim = EmbedDim;

  explicit InfoNceLoss(T temperature = static_cast<T>(1))
      : temperature_(temperature) {
    if (!(temperature > T{0})) {
      throw std::invalid_argument("temperature must be positive");
    }
  }

  T temperature() const { return temperature_; }

  // `candidates[positive]` is the matching document; the rest are negatives.
  T forward(const Vector& query, const std::vector<Vector>& candidates,
            std::size_t positive) {
    if (candidates.empty()) {
      throw std::invalid_argument("InfoNCE needs at least one candidate");
    }
    if (positive >= candidates.size()) {
      throw std::out_of_range("positive index " + std::to_string(positive) +
                              " outside [0, " +
                              std::to_string(candidates.size()) + ")");
    }

    const T inv_tau = T{1} / temperature_;
    const std::size_t n = candidates.size();

    probs_.assign(n, T{});
    T max_v = -std::numeric_limits<T>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
      probs_[i] = dot(query, candidates[i]) * inv_tau;
      if (probs_[i] > max_v) {
        max_v = probs_[i];
      }
    }
    T sum{};
    for (std::size_t i = 0; i < n; ++i) {
      probs_[i] = std::exp(probs_[i] - max_v);
      sum += probs_[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      probs_[i] /= sum;
    }

    // d(loss)/d(logit_i) = p_i - 1{i == positive}, then chain through the dot
    d_query_ = Vector{};
    d_candidates_.assign(n, Vector{});
    for (std::size_t i = 0; i < n; ++i) {
      const T g = (probs_[i] - (i == positive ? T{1} : T{0})) * inv_tau;
      for (std::size_t d = 0; d < EmbedDim; ++d) {
        d_query_(d) += g * candidates[i](d);
        d_candidates_[i](d) = g * query(d);
      }
    }
    ready_ = true;

    const T p = probs_[positive];
    const T floor = std::numeric_limits<T>::min();
    return -std::log(p > floor ? p : floor);
  }

  const Vector& query_grad() const {
    if (!ready_) {
      throw std::logic_error("backward() called before forward()");
    }
    return d_query_;
  }

  const Vector& candidate_grad(std::size_t i) const {
    if (!ready_) {
      throw std::logic_error("backward() called before forward()");
    }
    return d_candidates_.at(i);
  }

  // Softmax over candidates from the last forward().
  const std::vector<T>& probabilities() const { return probs_; }

  std::size_t argmax() const {
    std::size_t best = 0;
    for (std::size_t i = 1; i < probs_.size(); ++i) {
      if (probs_[i] > probs_[best]) {
        best = i;
      }
    }
    return best;
  }

 private:
  T temperature_;

  std::vector<T> probs_;
  Vector d_query_;
  std::vector<Vector> d_candidates_;
  bool ready_ = false;
};

}  // namespace zf
