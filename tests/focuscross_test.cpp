#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "focuscross.h"
#include "loss.h"
#include "lowrescross.h"
#include "replicas.h"

namespace {

// LowResCross's toy row (lowrescross_test.cpp): two groups of four values,
// either present or absent, each with four slots of state, and one learned
// id. Row: [values 8 | id 2 | state 8].
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

using Focus = zf::FocusCross<float, ToyLayout, 3, 4, 2, 4, 8>;
// In double for the finite differences.
using FocusD = zf::FocusCross<double, ToyLayout, 3, 4, 2, 4, 8>;

ToyRow random_row(std::mt19937& rng, bool both) {
  std::normal_distribution<float> dist(0.0f, 0.7f);
  ToyRow r;
  r.active = {true, both};
  for (float& v : r.values) v = dist(rng);
  for (float& v : r.state) v = dist(rng);
  r.id = static_cast<int>(rng() % ToyLayout::kIds);
  return r;
}

struct Example {
  std::vector<ToyRow> queries, workspace;
};

Example random_example(std::mt19937& rng, std::size_t nq, std::size_t nm) {
  Example e;
  for (std::size_t i = 0; i < nq; ++i) e.queries.push_back(random_row(rng, i % 2 == 0));
  for (std::size_t j = 0; j < nm; ++j) e.workspace.push_back(random_row(rng, j % 3 != 1));
  return e;
}

// A loss on both outputs: the readouts, and the final workspace rows (as a
// pointer would read them). Three levels, so a relevance set at one level
// reaches the loss through the next two.
struct Objective {
  FocusD::Hidden wh;
  std::vector<double> wm;  // over memory_size x kStream

  double value(const FocusD& f) const {
    double l = 0.0;
    const FocusD::Hidden h = f.readout_of(0);
    for (std::size_t i = 0; i < h.size(); ++i) l += h.data()[i] * wh.data()[i];
    const double* m = f.memory_rows(0);
    for (std::size_t i = 0; i < f.memory_size(0) * FocusD::kStream; ++i) l += m[i] * wm[i];
    return l;
  }
};

TEST(FocusCross, GradientsMatchFiniteDifferences) {
  std::mt19937 rng(11);
  FocusD f(3);
  f.randomize(rng);
  Example ex = random_example(rng, 3, 4);
  Objective obj;
  std::normal_distribution<double> dist(0.0, 1.0);
  for (std::size_t i = 0; i < obj.wh.size(); ++i) obj.wh.data()[i] = dist(rng);
  obj.wm.resize(4 * FocusD::kStream);
  for (double& v : obj.wm) v = dist(rng);
  const auto loss = [&] {
    f.forward(ex.queries, ex.workspace);
    return obj.value(f);
  };

  f.zero_grad();
  loss();
  for (std::size_t i = 0; i < f.memory_size(0) * FocusD::kStream; ++i) f.memory_grad(0)[i] += obj.wm[i];
  const auto d_state = f.backward(obj.wh);

  // Every parameter tensor, a few elements of each.
  std::vector<double*> params;
  std::vector<double> analytic;
  std::vector<std::vector<double>> grads;
  zf::visit_grads(f, [&](auto& t) { grads.emplace_back(t.data(), t.data() + t.size()); });
  std::size_t tensor = 0;
  zf::visit_params(f, [&](auto& t) {
    const std::size_t n = t.size(), step = std::max<std::size_t>(1, n / 5);
    for (std::size_t i = tensor % step; i < n; i += step) {
      params.push_back(t.data() + i);
      analytic.push_back(grads[tensor][i]);
    }
    ++tensor;
  });
  ASSERT_EQ(tensor, grads.size());
  const double h = 1e-5;
  for (std::size_t s = 0; s < params.size(); ++s) {
    const double v = *params[s];
    *params[s] = v + h;
    const double up = loss();
    *params[s] = v - h;
    const double down = loss();
    *params[s] = v;
    const double num = (up - down) / (2 * h);
    EXPECT_NEAR(analytic[s], num, 1e-4 * std::max(1.0, std::abs(num))) << "parameter slot " << s;
  }

  // And the input state of the query rows.
  for (std::size_t q = 0; q < ex.queries.size(); ++q) {
    for (std::size_t k = 0; k < ToyLayout::kState; ++k) {
      float& slot = ex.queries[q].state[k];
      const float v = slot;
      const float hf = 1e-3f;
      slot = v + hf;
      const double up = loss();
      slot = v - hf;
      const double down = loss();
      slot = v;
      const double num = (up - down) / (2 * hf);
      const bool active = ex.queries[q].active[k / 4];
      if (active) EXPECT_NEAR(d_state[q][k], num, 1e-3 * std::max(1.0, std::abs(num))) << q << " " << k;
    }
  }
}

// The relevance is what the queries looked at: in [0, 1), and every row
// gets some, since attention never quite leaves a key out.
TEST(FocusCross, RelevanceIsTheAttentionRowsGot) {
  std::mt19937 rng(3);
  Focus f(2);
  f.randomize(rng);
  const Example ex = random_example(rng, 3, 4);
  f.forward(ex.queries, ex.workspace);
  for (std::size_t level = 0; level < 2; ++level) {
    const float* r = f.relevance(0, level);
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_GT(r[j], 0.0f);
      EXPECT_LT(r[j], 1.0f);
    }
  }
}

// The workspace evolves only where its rows may write: an absent group's
// state stays absent, and the fixed part never changes.
TEST(FocusCross, TheWorkspaceWritesOnlyItsState) {
  std::mt19937 rng(5);
  Focus f(2);
  f.randomize(rng);
  const Example ex = random_example(rng, 3, 4);
  f.forward(ex.queries, ex.workspace);
  bool moved = false;
  for (std::size_t j = 0; j < 4; ++j) {
    const float* row = f.memory_rows(0) + j * ToyLayout::kWidth;
    const ToyRow& t = ex.workspace[j];
    for (std::size_t g = 0; g < 2; ++g) {
      for (std::size_t k = 0; k < 4; ++k) {
        EXPECT_EQ(row[g * 4 + k], t.active[g] ? t.values[g * 4 + k] : 0.0f);
        const float s = row[ToyLayout::kStateOffset + g * 4 + k];
        if (!t.active[g]) EXPECT_EQ(s, 0.0f);
        moved = moved || (t.active[g] && s != t.state[g * 4 + k]);
      }
    }
  }
  EXPECT_TRUE(moved);
}

// Passes never see each other: a batch gives each pass what it gives alone,
// a pass with no workspace among them.
TEST(FocusCross, PassesAreIndependent) {
  std::mt19937 rng(7);
  Focus f(2);
  f.randomize(rng);
  const Example a = random_example(rng, 3, 4), b = random_example(rng, 2, 0), c = random_example(rng, 1, 2);
  f.forward_batch({{&a.queries, &a.workspace}, {&b.queries, &b.workspace}, {&c.queries, &c.workspace}});
  const Focus::Hidden ha = f.readout_of(0), hb = f.readout_of(1), hc = f.readout_of(2);
  const std::vector<float> rc(f.relevance(2), f.relevance(2) + 2);
  for (const auto& [ex, h] : {std::pair{&a, &ha}, std::pair{&b, &hb}, std::pair{&c, &hc}}) {
    const Focus::Hidden& alone = f.forward(ex->queries, ex->workspace);
    for (std::size_t i = 0; i < alone.size(); ++i) EXPECT_NEAR(alone.data()[i], h->data()[i], 1e-5f) << i;
  }
  for (std::size_t j = 0; j < 2; ++j) EXPECT_NEAR(f.relevance()[j], rc[j], 1e-6f);
}

// A pointer reads the final workspace, and its gradient reaches the stack.
// The workspace's state is the stack's to write, so the pointer's gradient
// moves what it scores: a step of 0.05 diverges here unclipped (LowResCross,
// whose memory rows are fixed, takes it).
TEST(FocusCross, APointerTrainsThroughTheWorkspace) {
  std::mt19937 rng(9);
  Focus f(2);
  f.randomize(rng);
  zf::LowResPointer<float, ToyLayout, 4, 4> pointer;
  pointer.randomize(rng);
  std::vector<Example> data;
  std::vector<std::size_t> target;
  for (int n = 0; n < 16; ++n) {
    data.push_back(random_example(rng, 2, 4));
    // The target: the workspace row whose first value is largest.
    std::size_t best = 0;
    for (std::size_t j = 1; j < 4; ++j) {
      if (data.back().workspace[j].values[0] > data.back().workspace[best].values[0]) best = j;
    }
    target.push_back(best);
  }
  const auto epoch = [&](bool learn) {
    double total = 0.0;
    for (std::size_t n = 0; n < data.size(); ++n) {
      f.forward(data[n].queries, data[n].workspace);
      const auto r = f.readout(0, 0);
      const auto logits = pointer.forward(f, r);
      double top = logits(0);
      for (std::size_t j = 1; j < 4; ++j) top = std::max<double>(top, logits(j));
      double sum = 0.0;
      for (std::size_t j = 0; j < 4; ++j) sum += std::exp(logits(j) - top);
      total += -(logits(target[n]) - top - std::log(sum));
      if (!learn) continue;
      zf::Tensor<float, 4> d;
      for (std::size_t j = 0; j < 4; ++j) {
        d(j) = static_cast<float>(std::exp(logits(j) - top) / sum) - (j == target[n] ? 1.0f : 0.0f);
      }
      f.zero_grad();
      pointer.zero_grad();
      const auto d_r = pointer.backward(f, r, d);
      Focus::Hidden dh{};
      for (std::size_t k = 0; k < ToyLayout::kState; ++k) dh(0, k) = d_r(k);
      f.backward(dh);
      f.apply_sgd(0.01f);
      pointer.apply_sgd(0.01f);
    }
    return total / data.size();
  };
  const double before = epoch(false);
  for (int e = 0; e < 30; ++e) epoch(true);
  EXPECT_LT(epoch(false), 0.1 * before);
}

#if defined(ZTENSOR_APPLE_GPU)

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

// Three passes: full, with no workspace, and short (padding everywhere).
struct GpuCase {
  Focus cpu{3}, gpu{3};
  std::vector<Example> ex;
  std::vector<zf::LowResPass<ToyRow>> passes;

  GpuCase() {
    std::mt19937 rng(13);
    cpu.randomize(rng);
    gpu = cpu;
    cpu.set_device(zf::LowResDevice::kCpu);
    gpu.set_device(zf::LowResDevice::kApple);
    ex = {random_example(rng, 3, 4), random_example(rng, 2, 0), random_example(rng, 1, 2)};
    for (const Example& e : ex) passes.push_back({&e.queries, &e.workspace});
  }
};

// forward_batch() and backward_batch() on the GPU against the CPU: the
// readouts, the final workspace rows, every level's relevance, and every
// gradient, a pointer's on the workspace rows included.
TEST(FocusCross, ForwardAndBackwardOnTheGpuMatchTheCpu) {
  if (!zf::apple::available()) GTEST_SKIP() << "no Metal device";
  GpuCase c;
  c.cpu.forward_batch(c.passes);
  c.gpu.forward_batch(c.passes);
  EXPECT_EQ(c.gpu.ran_on(), zf::LowResDevice::kApple);
  std::mt19937 rng(17);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<Focus::Hidden> d(3);
  for (std::size_t b = 0; b < 3; ++b) {
    const Focus::Hidden hc = c.cpu.readout_of(b), hg = c.gpu.readout_of(b);
    for (std::size_t i = 0; i < hc.size(); ++i) EXPECT_NEAR(hg.data()[i], hc.data()[i], 1e-4f) << b << " " << i;
    const std::size_t nm = c.ex[b].workspace.size();
    ASSERT_EQ(c.gpu.memory_size(b), nm);
    for (std::size_t i = 0; i < nm * ToyLayout::kWidth; ++i) {
      EXPECT_NEAR(c.gpu.memory_rows(b)[i], c.cpu.memory_rows(b)[i], 1e-4f) << b << " " << i;
    }
    for (std::size_t level = 0; level < 3; ++level) {
      for (std::size_t j = 0; j < nm; ++j) {
        EXPECT_NEAR(c.gpu.relevance(b, level)[j], c.cpu.relevance(b, level)[j], 1e-5f) << b << " " << level;
      }
    }
    for (std::size_t i = 0; i < c.ex[b].queries.size() * ToyLayout::kState; ++i) d[b].data()[i] = dist(rng);
    for (std::size_t i = 0; i < nm * ToyLayout::kWidth; ++i) {
      const float v = dist(rng);
      c.cpu.memory_grad(b)[i] += v;
      c.gpu.memory_grad(b)[i] += v;
    }
  }
  c.cpu.zero_grad();
  c.gpu.zero_grad();
  const auto sc = c.cpu.backward_batch(d);
  const auto sg = c.gpu.backward_batch(d);
  expect_same_grads(grads_of(c.cpu), grads_of(c.gpu));
  for (std::size_t b = 0; b < 3; ++b) {
    for (std::size_t i = 0; i < sc[b].size(); ++i) {
      for (std::size_t k = 0; k < ToyLayout::kState; ++k) EXPECT_NEAR(sg[b][i][k], sc[b][i][k], 1e-3f) << b << " " << i;
    }
  }
}

// What a model puts on top (as in LowResCross's GPU test): a linear head on
// readout 0 with squared error, and a pointer from readout 1 into the final
// workspace rows with cross entropy.
struct ToyHeads {
  using Node = zf::apple::Node;
  static constexpr int kM = 4;
  zf::Linear<float, ToyLayout::kState, 3> head;
  zf::LowResPointer<float, ToyLayout, kM, 4> pointer;
  std::vector<std::array<float, 3>> y;
  std::vector<std::size_t> target, items;
  std::vector<std::array<float, 3>> got;

  template <typename Visit>
  void visit_layers(Visit&& visit) {
    visit(head);
    visit(pointer);
  }
  struct Nodes {
    Node live, pointed, y, hot, bias, out;  // pointed: the pass has a workspace
  };
  Node build(zf::LowResHeadsGraph& hg, Nodes* n) {
    zf::apple::Graph& g = hg.g;
    const int b = hg.batch, h = static_cast<int>(ToyLayout::kState);
    const Node hw = hg.params.next(), hb = hg.params.next();
    const Node aw = hg.params.next(), ab = hg.params.next(), bw = hg.params.next(), bb = hg.params.next();
    const Node h3 = g.reshape(hg.hidden, {b, hg.q, h});
    const Node r0 = g.reshape(g.slice(h3, 1, 0, 1), {b, h}), r1 = g.reshape(g.slice(h3, 1, 1, 1), {b, h});
    n->live = hg.in.add(g, {b, 1});
    n->pointed = hg.in.add(g, {b, 1});
    n->y = hg.in.add(g, {b, 3});
    n->hot = hg.in.add(g, {b, hg.m});
    n->bias = hg.in.add(g, {b, hg.m});
    n->out = zf::apple::linear(g, r0, hw, hb);
    hg.out.add(g, n->out);
    const Node mse = g.sum_all(g.mul(n->live, g.sum(g.square(g.sub(n->out, n->y)), {-1})));
    const Node qa = g.reshape(zf::apple::linear(g, r1, aw, ab), {b, 1, 4});
    const Node keys = g.reshape(zf::apple::linear(g, hg.rows, bw, bb), {b, hg.m, 4});
    const Node logits = g.reshape(g.matmul(qa, g.transpose(keys, 1, 2)), {b, hg.m});
    const Node ce = g.sum_all(g.mul(n->pointed, g.softmax_ce(g.add(logits, n->bias), n->hot)));
    return g.add(mse, ce);
  }
  void fill(const Nodes& n, zf::LowResGraphInputs& in) const {
    for (std::size_t b = 0; b < y.size(); ++b) {
      in[n.live].data()[b] = 1.0f;
      in[n.pointed].data()[b] = items[b] > 0 ? 1.0f : 0.0f;
      for (std::size_t k = 0; k < 3; ++k) in[n.y].data()[b * 3 + k] = y[b][k];
      if (items[b] > 0) in[n.hot].data()[b * kM + target[b]] = 1.0f;
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

// step_batch() -- the stack, the heads and every gradient in one GPU run --
// against the CPU's forward_batch(), the heads and backward_batch().
TEST(FocusCross, AStepOnTheGpuMatchesTheCpu) {
  if (!zf::apple::available()) GTEST_SKIP() << "no Metal device";
  GpuCase c;
  std::mt19937 rng(19);
  ToyHeads hc;
  hc.head.randomize(rng);
  hc.pointer.randomize(rng);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (std::size_t b = 0; b < 3; ++b) {
    const std::size_t nm = c.ex[b].workspace.size();
    hc.y.push_back({dist(rng), dist(rng), dist(rng)});
    hc.target.push_back(nm > 0 ? b % nm : 0);
    hc.items.push_back(nm);
  }
  ToyHeads hg = hc;
  c.cpu.zero_grad();
  c.gpu.zero_grad();
  hc.visit_layers([](auto& l) { l.zero_grad(); });
  hg.visit_layers([](auto& l) { l.zero_grad(); });

  c.cpu.forward_batch(c.passes);
  std::vector<Focus::Hidden> d(3);
  std::vector<std::array<float, 3>> want(3);
  for (std::size_t b = 0; b < 3; ++b) {
    const Focus::Hidden h = c.cpu.readout_of(b);
    Focus::Readout r0, r1;
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
    const auto g0 = hc.head.backward(r0, d_out);
    for (std::size_t k = 0; k < ToyLayout::kState; ++k) d[b](0, k) = g0(k);
    if (hc.items[b] == 0) continue;
    const auto logits = hc.pointer.forward(c.cpu, r1, b);
    zf::Tensor<float, ToyHeads::kM> d_logits{};
    zf::softmax_ce(logits, hc.target[b], hc.items[b], &d_logits);
    const auto g1 = hc.pointer.backward(c.cpu, r1, d_logits, b);
    for (std::size_t k = 0; k < ToyLayout::kState; ++k) d[b](1, k) = g1(k);
  }
  c.cpu.backward_batch(d);

  c.gpu.step_batch(c.passes, hg);
  EXPECT_EQ(c.gpu.ran_on(), zf::LowResDevice::kApple);
  for (std::size_t b = 0; b < 3; ++b) {
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(hg.got[b][k], want[b][k], 1e-4) << b;
  }
  expect_same_grads(grads_of(c.cpu), grads_of(c.gpu));
  expect_same_grads(grads_of(hc), grads_of(hg));
  EXPECT_THROW(c.gpu.backward_batch(d), std::logic_error);
}

#endif  // ZTENSOR_APPLE_GPU

}  // namespace
