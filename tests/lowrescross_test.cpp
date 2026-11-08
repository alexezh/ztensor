#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "grad_check.h"
#include "lowrescross.h"

namespace {

// A toy row: two groups of four values, either present or absent, each with
// four slots of state, and one learned id. Row: [values 8 | id 2 | state 8].
struct ToyRow {
  std::array<float, 8> values{};
  std::array<bool, 2> active{};
  std::array<float, 8> state{};
  int id = 0;
};

struct ToyLayout {
  using Input = ToyRow;
  using StateGrad = std::array<float, 8>;
  static constexpr std::size_t kWidth = 18;
  static constexpr std::size_t kStateOffset = 10;
  static constexpr std::size_t kState = 8;
  static constexpr std::size_t kIds = 3;
  static constexpr std::size_t kIdDim = 2;
  static constexpr std::size_t kIdSlots = 1;
  static constexpr std::array<std::size_t, kIdSlots> kIdAt = {8};

  template <typename T>
  static void put(T* row, const ToyRow& t) {
    for (std::size_t g = 0; g < 2; ++g) {
      if (!t.active[g]) continue;
      for (std::size_t k = 0; k < 4; ++k) {
        row[g * 4 + k] = t.values[g * 4 + k];
        row[kStateOffset + g * 4 + k] = t.state[g * 4 + k];
      }
    }
  }
  template <typename T>
  static void put_mask(T* mask, const ToyRow& t) {
    mask[8] = mask[9] = T{1};
    for (std::size_t g = 0; g < 2; ++g) {
      if (!t.active[g]) continue;
      for (std::size_t k = 0; k < 4; ++k) mask[g * 4 + k] = mask[kStateOffset + g * 4 + k] = T{1};
    }
  }
  template <typename T>
  static void put_write_mask(T* mask, const ToyRow& t) {
    for (std::size_t g = 0; g < 2; ++g) {
      if (!t.active[g]) continue;
      for (std::size_t k = 0; k < 4; ++k) mask[g * 4 + k] = T{1};
    }
  }
  static int id(const ToyRow& t, std::size_t) { return t.id; }
  template <typename T>
  static StateGrad state_grad(const T* d, const ToyRow&) {
    StateGrad g{};
    for (std::size_t k = 0; k < kState; ++k) g[k] = static_cast<float>(d[k]);
    return g;
  }
};

using Cross = zf::LowResCross<float, ToyLayout, 2, 3, 2, 2, 8>;
// In double for the finite differences.
using CrossD = zf::LowResCross<double, ToyLayout, 2, 3, 2, 2, 8>;

ToyRow random_row(std::mt19937& rng, bool both) {
  std::normal_distribution<float> dist(0.0f, 0.7f);
  ToyRow r;
  r.active = {true, both};
  for (float& v : r.values) v = dist(rng);
  for (float& v : r.state) v = dist(rng);
  r.id = static_cast<int>(rng() % ToyLayout::kIds);
  return r;
}

// A row's absent state stays absent: the layers restore what they compute
// only into the state the write mask allows.
TEST(LowResCross, AbsentStateStaysAbsent) {
  std::mt19937 rng(3);
  Cross cross(2);
  cross.randomize(rng);
  std::vector<ToyRow> queries = {random_row(rng, false), random_row(rng, true)};
  const std::vector<ToyRow> memory = {random_row(rng, true), random_row(rng, false), random_row(rng, true)};
  const auto& h = cross.forward(queries, memory);
  bool wrote = false;
  for (std::size_t k = 0; k < 4; ++k) {
    EXPECT_EQ(h(0, 4 + k), 0.0f) << k;  // query 0's second group is absent
    wrote = wrote || h(0, k) != queries[0].state[k];
  }
  EXPECT_TRUE(wrote);
}

// backward() hands back the gradient on each query row's input state.
TEST(LowResCross, InputStateGradientMatchesFiniteDifferences) {
  std::mt19937 rng(5);
  CrossD cross(2);
  cross.randomize(rng);
  std::vector<ToyRow> queries = {random_row(rng, true), random_row(rng, true)};
  const std::vector<ToyRow> memory = {random_row(rng, true), random_row(rng, false), random_row(rng, true)};
  CrossD::Hidden weight;
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (std::size_t i = 0; i < weight.size(); ++i) weight.data()[i] = dist(rng);
  const auto loss = [&] {
    const auto& h = cross.forward(queries, memory);
    double l = 0.0;
    for (std::size_t i = 0; i < h.size(); ++i) l += h.data()[i] * weight.data()[i];
    return static_cast<float>(l);
  };
  loss();
  const auto grads = cross.backward(weight);
  std::vector<float*> slots;
  std::vector<double> analytic;
  for (std::size_t q = 0; q < queries.size(); ++q) {
    for (std::size_t k = 0; k < ToyLayout::kState; ++k) {
      slots.push_back(&queries[q].state[k]);
      analytic.push_back(grads[q][k]);
    }
  }
  // A step small enough not to cross a ReLU kink.
  zf_test::expect_grads_match(slots, analytic, loss, "input state", 2e-2, 2e-4f);
}

// The cache's K and V follow the weights: after a step, a pass over memories
// cached before it reads what a batch made afresh would.
TEST(LowResCross, TheCacheFollowsTheWeights) {
  std::mt19937 rng(7);
  Cross cross(2);
  cross.randomize(rng);
  const std::vector<ToyRow> queries = {random_row(rng, true), random_row(rng, false)};
  const std::vector<ToyRow> memory = {random_row(rng, true), random_row(rng, false), random_row(rng, true)};
  cross.set_memories({&memory});
  cross.forward_shared({&queries}, {0});
  const Cross::Hidden before = cross.readout_of(0);
  // A step: every weight moves.
  std::normal_distribution<float> dist(0.0f, 0.05f);
  cross.visit_layers([&](auto& layer) {
    zf::visit_params(layer, [&](auto& t) {
      for (std::size_t i = 0; i < t.size(); ++i) t.data()[i] += dist(rng);
    });
  });
  cross.forward_shared({&queries}, {0});
  const Cross::Hidden cached = cross.readout_of(0);
  const Cross::Hidden& fresh = cross.forward(queries, memory);
  bool moved = false;
  for (std::size_t i = 0; i < fresh.size(); ++i) {
    EXPECT_NEAR(cached.data()[i], fresh.data()[i], 1e-5f) << i;
    moved = moved || std::abs(before.data()[i] - fresh.data()[i]) > 1e-3f;
  }
  EXPECT_TRUE(moved);
}

}  // namespace
