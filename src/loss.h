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
  static_assert(is_real_v<T>,
                "losses require a floating point element type");

 public:
  using value_type = T;
  using Acc = accumulator_t<T>;  // the loss itself is in the math type
  using Prediction = Tensor<T, Dims...>;

  Acc forward(const Prediction& pred, const Prediction& target) {
    Acc sum{};
    for (std::size_t i = 0; i < pred.size(); ++i) {
      const Acc d = pred[i] - target[i];
      d_pred_[i] = static_cast<T>(static_cast<Acc>(2) * d / static_cast<Acc>(pred.size()));
      sum += d * d;
    }
    ready_ = true;
    return sum / static_cast<Acc>(pred.size());
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
  static_assert(is_real_v<T>,
                "losses require a floating point element type");
  static_assert(NumClasses > 1, "cross entropy needs at least two classes");

 public:
  using value_type = T;
  using Acc = accumulator_t<T>;
  using Logits = Tensor<T, NumClasses>;

  static constexpr std::size_t num_classes = NumClasses;

  // Throws std::out_of_range when `label` is not a valid class index.
  Acc forward(const Logits& logits, std::size_t label) {
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
    d_logits_(label) = static_cast<T>(d_logits_(label) - Acc{1});
    ready_ = true;

    const Acc p = probs_(label);
    const Acc floor = std::numeric_limits<Acc>::min();
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
  static_assert(is_real_v<T>,
                "losses require a floating point element type");

 public:
  using value_type = T;
  using Acc = accumulator_t<T>;
  using Vector = Tensor<T, EmbedDim>;

  static constexpr std::size_t embed_dim = EmbedDim;

  explicit InfoNceLoss(Acc temperature = static_cast<Acc>(1))
      : temperature_(temperature) {
    if (!(temperature > Acc{0})) {
      throw std::invalid_argument("temperature must be positive");
    }
  }

  Acc temperature() const { return temperature_; }

  // `candidates[positive]` is the matching document; the rest are negatives.
  Acc forward(const Vector& query, const std::vector<Vector>& candidates,
            std::size_t positive) {
    if (candidates.empty()) {
      throw std::invalid_argument("InfoNCE needs at least one candidate");
    }
    if (positive >= candidates.size()) {
      throw std::out_of_range("positive index " + std::to_string(positive) +
                              " outside [0, " +
                              std::to_string(candidates.size()) + ")");
    }

    const Acc inv_tau = Acc{1} / temperature_;
    const std::size_t n = candidates.size();

    probs_.assign(n, Acc{});
    Acc max_v = -std::numeric_limits<Acc>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
      probs_[i] = dot(query, candidates[i]) * inv_tau;
      if (probs_[i] > max_v) {
        max_v = probs_[i];
      }
    }
    Acc sum{};
    for (std::size_t i = 0; i < n; ++i) {
      probs_[i] = std::exp(probs_[i] - max_v);
      sum += probs_[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      probs_[i] /= sum;
    }

    // d(loss)/d(logit_i) = p_i - 1{i == positive}, then chain through the dot
    Acc d_query[EmbedDim] = {};
    d_candidates_.assign(n, Vector{});
    for (std::size_t i = 0; i < n; ++i) {
      const Acc g = (probs_[i] - (i == positive ? Acc{1} : Acc{0})) * inv_tau;
      for (std::size_t d = 0; d < EmbedDim; ++d) {
        d_query[d] += g * candidates[i](d);
        d_candidates_[i](d) = static_cast<T>(g * query(d));
      }
    }
    for (std::size_t d = 0; d < EmbedDim; ++d) {
      d_query_(d) = static_cast<T>(d_query[d]);
    }
    ready_ = true;

    const Acc p = probs_[positive];
    const Acc floor = std::numeric_limits<Acc>::min();
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
  const std::vector<Acc>& probabilities() const { return probs_; }

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
  Acc temperature_;

  std::vector<Acc> probs_;
  Vector d_query_;
  std::vector<Vector> d_candidates_;
  bool ready_ = false;
};

}  // namespace zf
