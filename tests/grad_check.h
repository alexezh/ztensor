#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tensor.h"

// Finite-difference gradient checking. Every backward() in this library is
// verified against central differences of a scalar objective, which is the only
// way to catch a sign slip or a missing term in hand-written backprop.
namespace zf_test {

// Central difference d(loss)/d(*slot).
template <typename Loss>
double numeric_grad(Loss&& loss, float* slot, float h = 1e-3f) {
  const float original = *slot;
  *slot = original + h;
  const double up = static_cast<double>(loss());
  *slot = original - h;
  const double down = static_cast<double>(loss());
  *slot = original;
  return (up - down) / (2.0 * static_cast<double>(h));
}

// Compares an analytic gradient against the numeric one for every slot.
// `slots` are the scalars to perturb, `analytic` the matching backprop values.
inline void expect_grads_match(const std::vector<float*>& slots,
                               const std::vector<double>& analytic,
                               const std::function<float()>& loss,
                               const std::string& what, double tol = 2e-2,
                               float h = 1e-3f) {
  ASSERT_EQ(slots.size(), analytic.size()) << what;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const double num = numeric_grad(loss, slots[i], h);
    const double scale = std::max(1.0, std::max(std::abs(num),
                                                std::abs(analytic[i])));
    EXPECT_NEAR(analytic[i], num, tol * scale)
        << what << ": gradient mismatch at slot " << i;
  }
}

// Collects pointers to every element of a tensor, in flat order.
template <typename T, std::size_t... Dims>
std::vector<float*> slots_of(zf::Tensor<T, Dims...>& t) {
  std::vector<float*> out;
  out.reserve(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    out.push_back(&t[i]);
  }
  return out;
}

// Flattens a tensor into doubles, in flat order.
template <typename T, std::size_t... Dims>
std::vector<double> values_of(const zf::Tensor<T, Dims...>& t) {
  std::vector<double> out;
  out.reserve(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    out.push_back(static_cast<double>(t[i]));
  }
  return out;
}

// Fixed-seed generator so every gradient check is reproducible.
inline std::mt19937 rng(unsigned seed = 1234) { return std::mt19937(seed); }

template <typename T, std::size_t... Dims>
void randomize_tensor(zf::Tensor<T, Dims...>& t, std::mt19937& gen,
                      double stddev = 1.0) {
  std::normal_distribution<double> dist(0.0, stddev);
  for (std::size_t i = 0; i < t.size(); ++i) {
    t[i] = static_cast<T>(dist(gen));
  }
}

}  // namespace zf_test
