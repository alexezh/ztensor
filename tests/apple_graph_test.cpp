// zf::apple against the CPU library: each op's forward and its autodiff
// gradient against what ztensor computes (or a double-precision reference
// where ztensor has no single op for it).

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "apple/graph.h"
#include "apple/nn.h"
#include "apple/params.h"
#include "layers.h"
#include "loss.h"

namespace {

using zf::apple::Buffer;
using zf::apple::Graph;
using zf::apple::Node;

std::vector<float> random_values(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

Buffer buffer_of(const std::vector<float>& v) {
  Buffer b(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) b.data()[i] = v[i];
  return b;
}

void expect_near(const float* got, const float* want, std::size_t n, double tol, const char* what) {
  for (std::size_t i = 0; i < n; ++i) {
    const double scale = std::max(1.0, std::fabs(static_cast<double>(want[i])));
    ASSERT_NEAR(got[i], want[i], tol * scale) << what << " [" << i << "]";
  }
}

#define SKIP_WITHOUT_GPU() \
  if (!zf::apple::available()) GTEST_SKIP() << "no Metal device"

TEST(AppleGraph, RunsElementwiseAndReductions) {
  SKIP_WITHOUT_GPU();
  Graph g;
  const Node a = g.input({2, 3});
  const Node b = g.input({3});
  const Node y = g.sum(g.mul(g.add(a, b), g.constant(2.0)), {-1});
  Buffer ab = buffer_of({1, 2, 3, 4, 5, 6}), bb = buffer_of({10, 20, 30}), out(2);
  g.run({{a, ab}, {b, bb}}, {{y, out}});
  EXPECT_FLOAT_EQ(out.data()[0], 2 * (11 + 22 + 33));
  EXPECT_FLOAT_EQ(out.data()[1], 2 * (14 + 25 + 36));
  EXPECT_EQ(g.shape(y), (zf::apple::Shape{2, 1}));
}

TEST(AppleGraph, OneHotZeroesOutOfRangeIds) {
  SKIP_WITHOUT_GPU();
  Graph g;
  const Node ids = g.input_ints({4});
  const Node hot = g.one_hot(ids, 3);
  Buffer ib(4, Buffer::Type::kInt), out(12);
  const int v[4] = {0, 2, -1, 3};
  for (int i = 0; i < 4; ++i) ib.ints()[i] = v[i];
  g.run({{ids, ib}}, {{hot, out}});
  const float want[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 12; ++i) EXPECT_EQ(out.data()[i], want[i]) << i;
}

// gather picks rows, and its gradient adds into each row once per pick.
TEST(AppleGraph, GatherPicksRowsAndAddsTheirGradients) {
  SKIP_WITHOUT_GPU();
  Graph g;
  const Node a = g.input({3, 2});
  const Node ids = g.input_ints({4});
  const Node dy = g.input({4, 2});
  const Node y = g.gather(a, ids, 0);
  const Node da = g.gradients(g.sum_all(g.mul(y, dy)), {a})[0];
  Buffer ab = buffer_of({1, 2, 3, 4, 5, 6}), ib(4, Buffer::Type::kInt), db = buffer_of({1, 1, 2, 2, 3, 3, 4, 4});
  const int v[4] = {2, 0, 2, 1};
  for (int i = 0; i < 4; ++i) ib.ints()[i] = v[i];
  Buffer out(8), dout(6);
  g.run({{a, ab}, {ids, ib}, {dy, db}}, {{y, out}, {da, dout}});
  const float want[8] = {5, 6, 1, 2, 5, 6, 3, 4};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(out.data()[i], want[i]) << i;
  const float dwant[6] = {2, 2, 4, 4, 4, 4};
  for (int i = 0; i < 6; ++i) EXPECT_EQ(dout.data()[i], dwant[i]) << i;
}

// A tensor the loss does not depend on gets a zero gradient (MPSGraph
// itself would assert).
TEST(AppleGraph, UnusedInputsGetZeroGradients) {
  SKIP_WITHOUT_GPU();
  Graph g;
  const Node a = g.input({2});
  const Node unused = g.input({3});
  const std::vector<Node> d = g.gradients(g.sum_all(g.square(a)), {a, unused});
  Buffer ab = buffer_of({1, 2}), ub = buffer_of({5, 6, 7}), da(2), du = buffer_of({9, 9, 9});
  g.run({{a, ab}, {unused, ub}}, {{d[0], da}, {d[1], du}});
  EXPECT_FLOAT_EQ(da.data()[0], 2.0f);
  EXPECT_FLOAT_EQ(da.data()[1], 4.0f);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(du.data()[i], 0.0f) << i;
}

// y = linear(masked_layer_norm(x)), loss = sum(y * dy): forward and every
// gradient against zf::LayerNorm and zf::Linear.
TEST(AppleGraph, MaskedNormAndLinearMatchTheCpuLayers) {
  SKIP_WITHOUT_GPU();
  constexpr std::size_t kRows = 7, kIn = 24, kOut = 10;
  std::mt19937 rng(3);
  zf::LayerNorm<float, kIn> ln;
  zf::Linear<float, kIn, kOut> lin;
  std::mt19937_64 init(4);
  lin.randomize(init);
  for (std::size_t i = 0; i < kIn; ++i) {
    ln.gamma()(i) = 1.0f + 0.3f * std::sin(static_cast<float>(i));
    ln.beta()(i) = 0.2f * std::cos(static_cast<float>(i));
  }
  const std::vector<float> x = random_values(kRows * kIn, rng, 2.0f);
  std::vector<float> mask(kRows * kIn);
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t k = 0; k < kIn; ++k) mask[r * kIn + k] = (k + r) % 3 != 0 ? 1.0f : 0.0f;
  }
  // A row with nothing present comes out zero.
  for (std::size_t k = 0; k < kIn; ++k) mask[2 * kIn + k] = 0.0f;
  const std::vector<float> dy = random_values(kRows * kOut, rng);

  // CPU
  std::vector<zf::LayerNorm<float, kIn>::RowStats> stats(kRows);
  std::vector<float> xn(kRows * kIn), y(kRows * kOut), dxn(kRows * kIn, 0.0f), dx(kRows * kIn, 0.0f);
  ln.stats_rows(x.data(), mask.data(), kRows, stats.data());
  ln.forward_rows_from(stats.data(), x.data(), xn.data(), kRows, mask.data());
  lin.forward_into(xn.data(), y.data(), kRows);
  ln.zero_grad();
  lin.zero_grad();
  lin.backward_into(xn.data(), dy.data(), dxn.data(), kRows);
  ln.backward_rows_from(stats.data(), x.data(), dxn.data(), dx.data(), kRows, mask.data());

  // GPU
  Graph g;
  const Node xi = g.input({kRows, kIn}), mi = g.input({kRows, kIn}), dyi = g.input({kRows, kOut});
  const Node gamma = g.input({kIn}), beta = g.input({kIn}), w = g.input({kOut, kIn}), b = g.input({kOut});
  const Node yo = zf::apple::linear(g, zf::apple::masked_layer_norm(g, xi, mi, gamma, beta), w, b);
  const Node loss = g.sum_all(g.mul(yo, dyi));
  const std::vector<Node> d = g.gradients(loss, {xi, gamma, beta, w, b});
  Buffer gamma_b(kIn), beta_b(kIn), w_b(kOut * kIn), b_b(kOut);
  std::copy(ln.gamma().data(), ln.gamma().data() + kIn, gamma_b.data());
  std::copy(ln.beta().data(), ln.beta().data() + kIn, beta_b.data());
  std::copy(lin.weight().data(), lin.weight().data() + kOut * kIn, w_b.data());
  std::copy(lin.bias().data(), lin.bias().data() + kOut, b_b.data());
  Buffer y_b(kRows * kOut), dx_b(kRows * kIn), dg_b(kIn), db_b(kIn), dw_b(kOut * kIn), dbias_b(kOut);
  g.run({{xi, buffer_of(x)}, {mi, buffer_of(mask)}, {dyi, buffer_of(dy)}, {gamma, gamma_b}, {beta, beta_b},
         {w, w_b}, {b, b_b}},
        {{yo, y_b}, {d[0], dx_b}, {d[1], dg_b}, {d[2], db_b}, {d[3], dw_b}, {d[4], dbias_b}});

  expect_near(y_b.data(), y.data(), y.size(), 1e-4, "y");
  expect_near(dx_b.data(), dx.data(), dx.size(), 1e-3, "dx");
  expect_near(dg_b.data(), ln.gamma_grad().data(), kIn, 1e-3, "dgamma");
  expect_near(db_b.data(), ln.beta_grad().data(), kIn, 1e-3, "dbeta");
  expect_near(dw_b.data(), lin.weight_grad().data(), kOut * kIn, 1e-3, "dW");
  expect_near(dbias_b.data(), lin.bias_grad().data(), kOut, 1e-3, "db");
  for (std::size_t k = 0; k < kOut; ++k) EXPECT_NEAR(y_b.data()[2 * kOut + k], lin.bias()(k), 1e-6);
}

// attention() against a double-precision reference: the context, and the
// gradient on q, k and v by central differences of that reference.
TEST(AppleGraph, AttentionMatchesAReferenceWithPaddedKeys) {
  SKIP_WITHOUT_GPU();
  constexpr int kBatch = 2, kNq = 3, kNk = 4, kHeads = 2, kDim = 4, kHd = kHeads * kDim;
  const int valid[kBatch] = {4, 2};  // pass 1's last two keys are padding
  std::mt19937 rng(5);
  const std::vector<float> q = random_values(kBatch * kNq * kHd, rng);
  const std::vector<float> k = random_values(kBatch * kNk * kHd, rng);
  const std::vector<float> v = random_values(kBatch * kNk * kHd, rng);
  const std::vector<float> dc = random_values(kBatch * kNq * kHd, rng);
  std::vector<float> bias(kBatch * kNk, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int j = valid[b]; j < kNk; ++j) bias[b * kNk + j] = -1e9f;
  }
  // loss = sum(context * dc), in double.
  const auto reference = [&](const std::vector<double>& qd, const std::vector<double>& kd,
                             const std::vector<double>& vd, std::vector<double>* ctx) {
    double loss = 0.0;
    ctx->assign(q.size(), 0.0);
    for (int b = 0; b < kBatch; ++b) {
      for (int h = 0; h < kHeads; ++h) {
        for (int i = 0; i < kNq; ++i) {
          std::vector<double> p(valid[b]);
          double top = -1e300, sum = 0.0;
          for (int j = 0; j < valid[b]; ++j) {
            double s = 0.0;
            for (int d = 0; d < kDim; ++d) {
              s += qd[(b * kNq + i) * kHd + h * kDim + d] * kd[(b * kNk + j) * kHd + h * kDim + d];
            }
            p[j] = s / std::sqrt(static_cast<double>(kDim));
            top = std::max(top, p[j]);
          }
          for (double& x : p) sum += x = std::exp(x - top);
          for (int d = 0; d < kDim; ++d) {
            double c = 0.0;
            for (int j = 0; j < valid[b]; ++j) c += p[j] / sum * vd[(b * kNk + j) * kHd + h * kDim + d];
            const std::size_t at = (b * kNq + i) * kHd + h * kDim + d;
            (*ctx)[at] = c;
            loss += c * dc[at];
          }
        }
      }
    }
    return loss;
  };

  Graph g;
  const Node qi = g.input({kBatch * kNq, kHd}), ki = g.input({kBatch * kNk, kHd}), vi = g.input({kBatch * kNk, kHd});
  const Node bi = g.input({kBatch, 1, 1, kNk}), dci = g.input({kBatch * kNq, kHd});
  const Node ctx = zf::apple::attention(g, qi, ki, vi, bi, kBatch, kNq, kNk, kHeads);
  const std::vector<Node> d = g.gradients(g.sum_all(g.mul(ctx, dci)), {qi, ki, vi});
  Buffer ctx_b(q.size()), dq_b(q.size()), dk_b(k.size()), dv_b(v.size());
  g.run({{qi, buffer_of(q)}, {ki, buffer_of(k)}, {vi, buffer_of(v)}, {bi, buffer_of(bias)}, {dci, buffer_of(dc)}},
        {{ctx, ctx_b}, {d[0], dq_b}, {d[1], dk_b}, {d[2], dv_b}});

  std::vector<double> qd(q.begin(), q.end()), kd(k.begin(), k.end()), vd(v.begin(), v.end()), want;
  reference(qd, kd, vd, &want);
  for (std::size_t i = 0; i < want.size(); ++i) ASSERT_NEAR(ctx_b.data()[i], want[i], 1e-5) << "ctx " << i;
  const auto check = [&](std::vector<double>* x, const Buffer& grad, const char* what) {
    std::vector<double> scratch;
    for (std::size_t i = 0; i < x->size(); ++i) {
      const double keep = (*x)[i], h = 1e-5;
      (*x)[i] = keep + h;
      const double up = reference(qd, kd, vd, &scratch);
      (*x)[i] = keep - h;
      const double down = reference(qd, kd, vd, &scratch);
      (*x)[i] = keep;
      ASSERT_NEAR(grad.data()[i], (up - down) / (2 * h), 1e-4) << what << " " << i;
    }
  };
  check(&qd, dq_b, "dq");
  check(&kd, dk_b, "dk");
  check(&vd, dv_b, "dv");
}

TEST(AppleGraph, LossesMatchZfSoftmaxCeAndBce) {
  SKIP_WITHOUT_GPU();
  constexpr std::size_t kRows = 5, kC = 6;
  std::mt19937 rng(6);
  const std::vector<float> logits = random_values(kRows * kC, rng, 3.0f);
  const std::size_t n[kRows] = {6, 6, 3, 1, 6};  // classes that count
  const std::size_t label[kRows] = {0, 5, 2, 0, 3};
  std::vector<float> target(kRows * kC, 0.0f), bias(kRows * kC, 0.0f);
  for (std::size_t r = 0; r < kRows; ++r) {
    target[r * kC + label[r]] = 1.0f;
    for (std::size_t c = n[r]; c < kC; ++c) bias[r * kC + c] = -1e9f;
  }
  const std::vector<float> z = random_values(kRows, rng, 4.0f);
  const float y[kRows] = {1, 0, 1, 0, 1};

  Graph g;
  const Node li = g.input({kRows, kC}), ti = g.input({kRows, kC}), bi = g.input({kRows, kC});
  const Node zi = g.input({kRows}), yi = g.input({kRows});
  const Node ce = g.softmax_ce(g.add(li, bi), ti);
  const Node bce = zf::apple::bce_logits(g, zi, yi);
  const std::vector<Node> d = g.gradients(g.add(g.sum_all(ce), g.sum_all(bce)), {li, zi});
  Buffer ce_b(kRows), bce_b(kRows), dl_b(kRows * kC), dz_b(kRows);
  g.run({{li, buffer_of(logits)}, {ti, buffer_of(target)}, {bi, buffer_of(bias)}, {zi, buffer_of(z)},
         {yi, buffer_of(std::vector<float>(y, y + kRows))}},
        {{ce, ce_b}, {bce, bce_b}, {d[0], dl_b}, {d[1], dz_b}});

  for (std::size_t r = 0; r < kRows; ++r) {
    zf::Tensor<float, kC> lt, dl;
    for (std::size_t c = 0; c < kC; ++c) lt(c) = logits[r * kC + c];
    const double want = zf::softmax_ce(lt, label[r], n[r], &dl);
    EXPECT_NEAR(ce_b.data()[r], want, 1e-5) << r;
    for (std::size_t c = 0; c < kC; ++c) EXPECT_NEAR(dl_b.data()[r * kC + c], dl(c), 1e-6) << r << "," << c;
    float dz = 0.0f;
    EXPECT_NEAR(bce_b.data()[r], zf::bce(z[r], y[r] > 0.5f, &dz), 1e-5) << r;
    EXPECT_NEAR(dz_b.data()[r], dz, 1e-6) << r;
  }
}

// clipped_sgd's step: clipped when the gradient is long, plain when short.
TEST(AppleGraph, ClippedSgdMatchesTheBatchStep) {
  SKIP_WITHOUT_GPU();
  std::mt19937 rng(7);
  const std::vector<float> p0 = random_values(5, rng), p1 = random_values(3, rng);
  const std::vector<float> g0 = random_values(5, rng), g1 = random_values(3, rng);
  double norm = 0.0;
  for (float v : g0) norm += v * v;
  for (float v : g1) norm += v * v;
  norm = std::sqrt(norm);
  for (const double max_norm : {norm / 4, norm * 4}) {
    Graph g;
    const Node a = g.input({5}), b = g.input({3}), ga = g.input({5}), gb = g.input({3});
    const Node lr = g.input({1}), mx = g.input({1});
    const std::vector<Node> up = zf::apple::clipped_sgd(g, {a, b}, {ga, gb}, lr, mx);
    Buffer oa(5), ob(3);
    g.run({{a, buffer_of(p0)}, {b, buffer_of(p1)}, {ga, buffer_of(g0)}, {gb, buffer_of(g1)},
           {lr, buffer_of({0.1f})}, {mx, buffer_of({static_cast<float>(max_norm)})}},
          {{up[0], oa}, {up[1], ob}});
    const double step = 0.1 * (norm > max_norm ? max_norm / norm : 1.0);
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(oa.data()[i], p0[i] - step * g0[i], 1e-6);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(ob.data()[i], p1[i] - step * g1[i], 1e-6);
  }
}

// A model's parameters up and back, and steps that keep them on the GPU.
struct TinyModel {
  zf::Linear<float, 4, 3> lin;
  zf::LayerNorm<float, 4> ln;
  template <typename Visit>
  void visit_layers(Visit&& visit) {
    visit(ln);
    visit(lin);
  }
};

TEST(AppleGraph, ParamsTrainOnTheGpuAndComeBack) {
  SKIP_WITHOUT_GPU();
  TinyModel cpu, gpu;
  std::mt19937_64 init(8);
  cpu.lin.randomize(init);
  gpu = cpu;
  std::mt19937 rng(9);
  const std::vector<float> x = random_values(2 * 4, rng), dy = random_values(2 * 3, rng);
  const std::vector<float> mask(2 * 4, 1.0f);

  Graph g;
  zf::apple::Params params;
  params.declare(g, gpu);
  ASSERT_EQ(params.size(), 4u);
  const Node xi = g.input({2, 4}), mi = g.input({2, 4}), dyi = g.input({2, 3});
  const Node gamma = params.next(), beta = params.next(), w = params.next(), b = params.next();
  EXPECT_TRUE(params.all_taken());
  const Node y = zf::apple::linear(g, zf::apple::masked_layer_norm(g, xi, mi, gamma, beta), w, b);
  const Node loss = g.sum_all(g.mul(y, dyi));
  const std::vector<Node> grads = g.gradients(loss, params.nodes());
  const Node lr = g.input({1}), mx = g.input({1});
  const std::vector<Node> up = zf::apple::clipped_sgd(g, params.nodes(), grads, lr, mx);
  params.upload(gpu);
  for (int step = 0; step < 3; ++step) {
    std::vector<Graph::Bind> in{{xi, buffer_of(x)}, {mi, buffer_of(mask)}, {dyi, buffer_of(dy)},
                                {lr, buffer_of({0.05f})}, {mx, buffer_of({1e6f})}}, out;
    params.bind(&in);
    params.bind_updated(up, &out);
    g.run(in, out);
    params.flip();

    // The same step on the CPU.
    std::vector<zf::LayerNorm<float, 4>::RowStats> stats(2);
    std::vector<float> xn(8), yo(6), dxn(8, 0.0f), dx(8);
    cpu.ln.zero_grad();
    cpu.lin.zero_grad();
    cpu.ln.stats_rows(x.data(), mask.data(), 2, stats.data());
    cpu.ln.forward_rows_from(stats.data(), x.data(), xn.data(), 2, mask.data());
    cpu.lin.forward_into(xn.data(), yo.data(), 2);
    cpu.lin.backward_into(xn.data(), dy.data(), dxn.data(), 2);
    cpu.ln.backward_rows_from(stats.data(), x.data(), dxn.data(), dx.data(), 2, mask.data());
    cpu.ln.apply_sgd(0.05f);
    cpu.lin.apply_sgd(0.05f);
  }
  params.download(gpu);
  expect_near(gpu.lin.weight().data(), cpu.lin.weight().data(), 12, 1e-5, "W");
  expect_near(gpu.lin.bias().data(), cpu.lin.bias().data(), 3, 1e-5, "b");
  expect_near(gpu.ln.gamma().data(), cpu.ln.gamma().data(), 4, 1e-5, "gamma");
  expect_near(gpu.ln.beta().data(), cpu.ln.beta().data(), 4, 1e-5, "beta");
}

}  // namespace
