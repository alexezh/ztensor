#include "replicas.h"

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "layers.h"

namespace {

// A model small enough to check by hand: an embedding added to x, a layer
// norm, a linear map, squared error against a target.
struct Toy {
  struct Stats {
    double loss = 0.0;
    std::size_t examples = 0;
    Stats& operator+=(const Stats& o) {
      loss += o.loss;
      examples += o.examples;
      return *this;
    }
  };
  struct Example {
    std::size_t id;
    zf::Tensor<float, 8> x;
    zf::Tensor<float, 4> target;
  };

  zf::Embedding<float, 16, 8> embed;
  zf::LayerNorm<float, 8> norm;
  zf::Linear<float, 8, 4> lin;
  zf::Replicas<Toy> replicas;

  template <typename Visit>
  void visit_layers(Visit&& visit) {
    visit(embed);
    visit(norm);
    visit(lin);
  }

  // Forward and backward on examples [b0, b1), adding into the gradients.
  void slice(const std::vector<Example>& exs, std::size_t b0, std::size_t b1, Stats* stats) {
    for (std::size_t b = b0; b < b1; ++b) {
      const Example& ex = exs[b];
      zf::Tensor<float, 8> in = ex.x;
      const auto e = embed.forward(ex.id);
      for (std::size_t k = 0; k < 8; ++k) in(k) += e(k);
      const auto h = norm.forward(in);
      const auto y = lin.forward(h);
      zf::Tensor<float, 4> dy;
      for (std::size_t k = 0; k < 4; ++k) {
        dy(k) = y(k) - ex.target(k);
        stats->loss += 0.5 * dy(k) * dy(k);
      }
      const auto dh = lin.backward(h, dy);
      embed.backward(ex.id, norm.backward(in, dh));
      ++stats->examples;
    }
  }
};

std::vector<Toy::Example> examples(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<Toy::Example> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i].id = i % 16;
    for (float& v : out[i].x) v = g(rng);
    for (float& v : out[i].target) v = g(rng);
  }
  return out;
}

std::vector<float> weights(Toy& m) {
  std::vector<float> out;
  for (const auto& b : zf::param_blocks<float>(m)) out.insert(out.end(), b.data, b.data + b.size);
  return out;
}

// A step size that clips: what a trainer's rate might be.
float rate(double norm2) {
  const double norm = std::sqrt(norm2);
  return static_cast<float>(0.02 * (norm > 2.0 ? 2.0 / norm : 1.0));
}

// Sliced steps take the same steps as one model running the whole batch
// and stepping by hand: the helpers start each step from the model's
// weights, their gradients all reach it, and rate() sees the whole norm.
TEST(Replicas, SlicedStepsMatchOneModel) {
  std::mt19937_64 rng(3);
  Toy sliced;
  sliced.embed.randomize(rng, 0.5f);
  sliced.lin.randomize(rng);
  Toy whole = sliced;
  const auto start = weights(sliced);
  const auto exs = examples(10, 7);  // 4 slices: 3, 3, 3, 1
  for (int step = 0; step < 3; ++step) {
    Toy::Stats s1, s2;
    sliced.replicas.run(sliced, exs.size(), 4, true, &s1,
                        [&](Toy& m, std::size_t b0, std::size_t b1, Toy::Stats* s) {
                          m.slice(exs, b0, b1, s);
                        },
                        rate);
    whole.visit_layers([](auto& l) { l.zero_grad(); });
    whole.slice(exs, 0, exs.size(), &s2);
    const auto w = zf::param_blocks<float>(whole);
    const auto g = zf::grad_blocks<float>(whole);
    double norm2 = 0.0;
    for (const auto& b : g) {
      for (std::size_t j = 0; j < b.size; ++j) norm2 += static_cast<double>(b.data[j]) * b.data[j];
    }
    const float r = rate(norm2);
    for (std::size_t i = 0; i < w.size(); ++i) {
      for (std::size_t j = 0; j < w[i].size; ++j) w[i].data[j] -= r * g[i].data[j];
    }
    EXPECT_EQ(s1.examples, exs.size());
    EXPECT_NEAR(s1.loss, s2.loss, 1e-4 * s2.loss);
  }
  const auto a = weights(sliced), b = weights(whole);
  ASSERT_EQ(a.size(), b.size());
  double moved = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], b[i], 1e-5f) << i;
    moved += std::abs(a[i] - start[i]);
  }
  EXPECT_GT(moved, 0.05);
  // The step leaves no gradient behind, in the model or a helper.
  for (const auto& blk : zf::grad_blocks<float>(sliced)) {
    for (std::size_t j = 0; j < blk.size; ++j) ASSERT_EQ(blk.data[j], 0.0f);
  }
}

// Evaluation splits the same way, without touching the weights.
TEST(Replicas, EvaluationLeavesWeights) {
  std::mt19937_64 rng(5);
  Toy m;
  m.lin.randomize(rng);
  const auto before = weights(m);
  const auto exs = examples(9, 11);
  Toy::Stats s;
  m.replicas.run(m, exs.size(), 4, false, &s,
                 [&](Toy& r, std::size_t b0, std::size_t b1, Toy::Stats* st) { r.slice(exs, b0, b1, st); },
                 rate);
  EXPECT_EQ(s.examples, exs.size());
  EXPECT_EQ(weights(m), before);
}

// Weights changed outside run() (a load) reach the helpers before the next
// batch: a model with stale helpers would score differently.
TEST(Replicas, HelpersFollowWeightsChangedOutside) {
  std::mt19937_64 rng(9);
  Toy m;
  m.lin.randomize(rng);
  const auto exs = examples(8, 13);
  const auto eval = [&](Toy& model) {
    Toy::Stats s;
    model.replicas.run(model, exs.size(), 4, false, &s,
                       [&](Toy& r, std::size_t b0, std::size_t b1, Toy::Stats* st) {
                         r.slice(exs, b0, b1, st);
                       },
                       rate);
    return s.loss;
  };
  eval(m);  // makes the helpers
  m.lin.randomize(rng);
  Toy fresh = m;  // no helpers of its own
  EXPECT_DOUBLE_EQ(eval(m), eval(fresh));
}

}  // namespace
