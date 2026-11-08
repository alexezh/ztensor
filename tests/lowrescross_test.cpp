#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "grad_check.h"
#include "loss.h"
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

#if defined(ZTENSOR_APPLE_GPU)

// Heads for a whole step on the GPU: a linear head on query row 0 against
// a target (squared error), and a pointer from row 1 into the memory
// (softmax cross entropy over the pass's rows).
struct ToyHeads {
  using Node = zf::apple::Node;
  static constexpr int kM = 3;
  zf::Linear<float, ToyLayout::kState, 3> head;
  zf::LowResPointer<float, ToyLayout, kM, 4> pointer;
  // The batch's labels, per pass.
  std::vector<std::array<float, 3>> y;
  std::vector<std::size_t> target, items;
  std::vector<std::array<float, 3>> got;  // the head, read back

  template <typename Visit>
  void visit_layers(Visit&& visit) {
    visit(head);
    visit(pointer);
  }
  struct Nodes {
    Node live, y, hot, bias, out;
  };
  Node build(zf::LowResHeadsGraph& hg, Nodes* n) {
    zf::apple::Graph& g = hg.g;
    const int b = hg.batch, h = static_cast<int>(ToyLayout::kState);
    const Node hw = hg.params.next(), hb = hg.params.next();
    const Node aw = hg.params.next(), ab = hg.params.next(), bw = hg.params.next(), bb = hg.params.next();
    const Node h3 = g.reshape(hg.hidden, {b, hg.q, h});
    const Node r0 = g.reshape(g.slice(h3, 1, 0, 1), {b, h}), r1 = g.reshape(g.slice(h3, 1, 1, 1), {b, h});
    n->live = hg.in.add(g, {b, 1});
    n->y = hg.in.add(g, {b, 3});
    n->hot = hg.in.add(g, {b, hg.m});
    n->bias = hg.in.add(g, {b, hg.m});
    n->out = zf::apple::linear(g, r0, hw, hb);
    hg.out.add(g, n->out);
    const Node mse = g.sum_all(g.mul(n->live, g.sum(g.square(g.sub(n->out, n->y)), {-1})));
    const Node qa = g.reshape(zf::apple::linear(g, r1, aw, ab), {b, 1, 4});
    const Node keys = g.reshape(zf::apple::linear(g, hg.rows, bw, bb), {b, hg.m, 4});
    const Node logits = g.reshape(g.matmul(qa, g.transpose(keys, 1, 2)), {b, hg.m});
    const Node ce = g.sum_all(g.softmax_ce(g.add(logits, n->bias), n->hot));
    return g.add(mse, ce);
  }
  void fill(const Nodes& n, zf::LowResGraphInputs& in) const {
    for (std::size_t b = 0; b < y.size(); ++b) {
      in[n.live].data()[b] = 1.0f;
      for (std::size_t k = 0; k < 3; ++k) in[n.y].data()[b * 3 + k] = y[b][k];
      in[n.hot].data()[b * kM + target[b]] = 1.0f;
      for (std::size_t j = items[b]; j < kM; ++j) in[n.bias].data()[b * kM + j] = -1e9f;
    }
  }
  void read(const Nodes& n, const zf::LowResGraphOutputs& out) {
    got.resize(y.size());
    for (std::size_t b = 0; b < y.size(); ++b) {
      for (std::size_t k = 0; k < 3; ++k) got[b][k] = out[n.out].data()[b * 3 + k];
    }
  }
};

template <typename Model>
std::vector<std::vector<float>> grads_of(Model& model) {
  std::vector<std::vector<float>> out;
  zf::visit_grads(model, [&](auto& t) { out.emplace_back(t.data(), t.data() + t.size()); });
  return out;
}

void expect_same_grads(const std::vector<std::vector<float>>& want, const std::vector<std::vector<float>>& got) {
  ASSERT_EQ(want.size(), got.size());
  for (std::size_t t = 0; t < want.size(); ++t) {
    float top = 0.0f;
    for (float v : want[t]) top = std::max(top, std::fabs(v));
    for (std::size_t i = 0; i < want[t].size(); ++i) {
      ASSERT_NEAR(got[t][i], want[t][i], 2e-3 * std::max(1.0f, top)) << "tensor " << t << " [" << i << "]";
    }
  }
}

// step_batch() -- the stack, the heads and every gradient in one GPU run --
// against the CPU's forward_batch(), the heads and backward_batch().
TEST(LowResCross, AStepOnTheGpuMatchesTheCpu) {
  if (!zf::apple::available()) GTEST_SKIP() << "no Metal device";
  std::mt19937 rng(9);
  Cross cpu(2);
  cpu.randomize(rng);
  Cross gpu = cpu;
  cpu.set_device(zf::LowResDevice::kCpu);
  gpu.set_device(zf::LowResDevice::kApple);
  ToyHeads hc;
  hc.head.randomize(rng);
  hc.pointer.randomize(rng);
  ToyHeads hg = hc;
  // Three passes; the last with one memory row.
  std::vector<std::vector<ToyRow>> queries, memory;
  const std::size_t nm[3] = {3, 2, 1};
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (std::size_t b = 0; b < 3; ++b) {
    queries.push_back({random_row(rng, true), random_row(rng, b != 1)});
    memory.emplace_back();
    for (std::size_t j = 0; j < nm[b]; ++j) memory.back().push_back(random_row(rng, j % 2 == 0));
    hc.y.push_back({dist(rng), dist(rng), dist(rng)});
    hc.target.push_back(b % nm[b]);
    hc.items.push_back(nm[b]);
  }
  hg.y = hc.y;
  hg.target = hc.target;
  hg.items = hc.items;
  std::vector<zf::LowResPass<ToyRow>> passes;
  for (std::size_t b = 0; b < 3; ++b) passes.push_back({&queries[b], &memory[b]});
  for (auto* m : {&cpu, &gpu}) m->zero_grad();
  hc.visit_layers([](auto& l) { l.zero_grad(); });
  hg.visit_layers([](auto& l) { l.zero_grad(); });

  cpu.forward_batch(passes);
  std::vector<Cross::Hidden> d(3);
  std::vector<std::array<float, 3>> want(3);
  for (std::size_t b = 0; b < 3; ++b) {
    const Cross::Hidden h = cpu.readout_of(b);
    Cross::Readout r0, r1;
    for (std::size_t k = 0; k < ToyLayout::kState; ++k) {
      r0(k) = h(0, k);
      r1(k) = h(1, k);
    }
    const auto out = hc.head.forward(r0);
    zf::Tensor<float, 3> d_out;
    for (std::size_t k = 0; k < 3; ++k) {
      want[b][k] = out(k);
      d_out(k) = 2.0f * (out(k) - hc.y[b][k]);
    }
    const auto logits = hc.pointer.forward(cpu, r1, b);
    zf::Tensor<float, ToyHeads::kM> d_logits{};
    zf::softmax_ce(logits, hc.target[b], hc.items[b], &d_logits);
    const auto g0 = hc.head.backward(r0, d_out);
    const auto g1 = hc.pointer.backward(cpu, r1, d_logits, b);
    for (std::size_t k = 0; k < ToyLayout::kState; ++k) {
      d[b](0, k) = g0(k);
      d[b](1, k) = g1(k);
    }
  }
  cpu.backward_batch(d);

  gpu.step_batch(passes, hg);
  EXPECT_EQ(gpu.ran_on(), zf::LowResDevice::kApple);
  for (std::size_t b = 0; b < 3; ++b) {
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(hg.got[b][k], want[b][k], 1e-4) << b;
  }
  expect_same_grads(grads_of(cpu), grads_of(gpu));
  expect_same_grads(grads_of(hc), grads_of(hg));
  EXPECT_THROW(gpu.backward_batch(d), std::logic_error);
}

#endif  // ZTENSOR_APPLE_GPU

}  // namespace
