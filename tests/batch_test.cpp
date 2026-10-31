// Batching and storage. A batch must compute exactly what the same examples
// would one at a time: same outputs, same input gradients, and parameter
// gradients equal to the sum over the examples. These tests hold the batched
// paths to that, plus finite differences straight through a batched stack.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>

#include <gtest/gtest.h>

#include "attention.h"
#include "cross_attention.h"
#include "grad_check.h"
#include "layers.h"
#include "tensor.h"
#include "transformer.h"

namespace {

template <typename T, std::size_t... Dims>
void fill(zf::Tensor<T, Dims...>& t, std::mt19937& rng, float stddev = 0.7f) {
  std::normal_distribution<float> dist(0.0f, stddev);
  for (T& v : t) v = dist(rng);
}

// Row r of a row-stacked tensor.
template <std::size_t Rows, std::size_t Dim>
const float* row(const zf::Tensor<float, Rows, Dim>& t, std::size_t r) {
  return t.data() + r * Dim;
}

template <typename A, typename B>
void expect_tensors_near(const A& a, const B& b, double tol, const char* what) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double scale = std::max(1.0, std::abs(static_cast<double>(a[i])));
    EXPECT_NEAR(a[i], b[i], tol * scale) << what << " at " << i;
  }
}

constexpr double kTol = 1e-5;

// --- storage ---------------------------------------------------------------

TEST(TensorStorage, InlineAndHeapAreBothZeroedAndAligned) {
  zf::Tensor<float, 4, 4> small;                // inline
  zf::Tensor<float, 128, 64> large;             // 32 KB: heap
  static_assert(sizeof(small) < 1024);
  static_assert(sizeof(large) < 64);
  for (float v : small) EXPECT_EQ(v, 0.0f);
  for (float v : large) EXPECT_EQ(v, 0.0f);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(small.data()) % 64, 0u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(large.data()) % 64, 0u);
}

TEST(TensorStorage, HeapCopiesAreDeepAndMovesLeaveAUsableTensor) {
  zf::Tensor<float, 128, 64> a;
  a(3, 5) = 2.5f;
  zf::Tensor<float, 128, 64> b = a;
  b(3, 5) = 7.0f;
  EXPECT_EQ(a(3, 5), 2.5f);

  zf::Tensor<float, 128, 64> c = std::move(a);
  EXPECT_EQ(c(3, 5), 2.5f);
  a(0, 0) = 1.0f;  // moved-from: still a valid, writable tensor
  EXPECT_EQ(a(0, 0), 1.0f);

  b = std::move(c);
  EXPECT_EQ(b(3, 5), 2.5f);
  c.fill(4.0f);
  EXPECT_EQ(c(127, 63), 4.0f);
}

TEST(TensorStorage, AllFiniteSpotsNanAndInfinity) {
  zf::Tensor<float, 3, 3> t;
  EXPECT_TRUE(zf::all_finite(t));
  t(1, 1) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(zf::all_finite(t));
  t(1, 1) = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(zf::all_finite(t));
  EXPECT_TRUE(zf::all_finite(zf::Tensor<std::int8_t, 2>{}));
}

// --- Linear row blocking ---------------------------------------------------

// The 4-row kernel must match one row at a time exactly: same summation order
// per output, gradients accumulated row by row in order.
template <std::size_t In, std::size_t Out>
void check_row_blocking() {
  constexpr std::size_t kRows = 11;  // two blocks of 4 and a remainder of 3
  std::mt19937 rng(3);
  zf::Linear<float, In, Out> blocked;
  blocked.randomize(rng);
  zf::Linear<float, In, Out> single = blocked;
  zf::Tensor<float, kRows, In> x;
  zf::Tensor<float, kRows, Out> g;
  fill(x, rng);
  fill(g, rng);

  const auto out = blocked.template forward<kRows>(x);
  const auto dx = blocked.template backward<kRows>(x, g);
  for (std::size_t r = 0; r < kRows; ++r) {
    zf::Tensor<float, In> xr;
    zf::Tensor<float, Out> gr;
    for (std::size_t i = 0; i < In; ++i) xr(i) = x(r, i);
    for (std::size_t o = 0; o < Out; ++o) gr(o) = g(r, o);
    const auto yr = single.forward(xr);
    const auto dxr = single.backward(xr, gr);
    for (std::size_t o = 0; o < Out; ++o) EXPECT_EQ(out(r, o), yr(o)) << r;
    for (std::size_t i = 0; i < In; ++i) EXPECT_EQ(dx(r, i), dxr(i)) << r;
  }
  EXPECT_EQ(blocked.weight_grad(), single.weight_grad());
  EXPECT_EQ(blocked.bias_grad(), single.bias_grad());
}

TEST(LinearBlocking, FourRowKernelMatchesOneRowExactly) {
  check_row_blocking<32, 32>();
  check_row_blocking<6, 5>();  // leftover columns and leftover outputs
  check_row_blocking<3, 9>();  // narrower than a vector
}

// --- batched attention and blocks ------------------------------------------

constexpr std::size_t kB = 3;
constexpr std::size_t kSeq = 5;
constexpr std::size_t kMem = 6;
constexpr std::size_t kDim = 8;
constexpr std::size_t kFf = 16;
constexpr std::size_t kValid[kB] = {5, 3, 1};
constexpr std::size_t kMemValid[kB] = {6, 2, 4};

// Copies example b's rows out of / into a row-stacked batch tensor.
template <std::size_t Len, std::size_t BatchRows>
zf::Tensor<float, Len, kDim> slice(const zf::Tensor<float, BatchRows, kDim>& t,
                                   std::size_t b) {
  zf::Tensor<float, Len, kDim> out;
  std::copy(row(t, b * Len), row(t, b * Len) + Len * kDim, out.data());
  return out;
}

// Zero the rows of padded positions, so the incoming gradient looks like what
// a real loss produces: nothing for rows no one reads.
template <std::size_t Len, std::size_t BatchRows>
void zero_padding(zf::Tensor<float, BatchRows, kDim>& t, const std::size_t* valid) {
  for (std::size_t b = 0; b < kB; ++b) {
    for (std::size_t r = valid[b]; r < Len; ++r) {
      for (std::size_t d = 0; d < kDim; ++d) t(b * Len + r, d) = 0.0f;
    }
  }
}

template <std::size_t Len, std::size_t BatchRows>
void expect_valid_rows_match(const zf::Tensor<float, BatchRows, kDim>& batched,
                             const zf::Tensor<float, Len, kDim>& single,
                             std::size_t b, std::size_t valid, const char* what) {
  for (std::size_t r = 0; r < valid; ++r) {
    for (std::size_t d = 0; d < kDim; ++d) {
      EXPECT_NEAR(batched(b * Len + r, d), single(r, d), kTol)
          << what << " example " << b << " row " << r;
    }
  }
}

TEST(BatchedSelfAttention, MatchesOneExampleAtATime) {
  for (bool causal : {false, true}) {
    std::mt19937 rng(5);
    zf::MultiHeadSelfAttention<float, kSeq, kDim, kB> batched(2, causal);
    std::mt19937 init(11);
    batched.randomize(init);
    zf::MultiHeadSelfAttention<float, kSeq, kDim> single(2, causal);
    init.seed(11);
    single.randomize(init);

    zf::Tensor<float, kB * kSeq, kDim> x;
    zf::Tensor<float, kB * kSeq, kDim> g;
    fill(x, rng);
    fill(g, rng);
    zero_padding<kSeq>(g, kValid);
    for (std::size_t b = 0; b < kB; ++b) batched.set_valid_length(b, kValid[b]);
    const auto y = batched.forward(x);
    const auto dx = batched.backward(g);

    for (std::size_t b = 0; b < kB; ++b) {
      single.set_valid_length(kValid[b]);
      const auto yb = single.forward(slice<kSeq>(x, b));
      const auto dxb = single.backward(slice<kSeq>(g, b));  // grads accumulate
      expect_valid_rows_match<kSeq>(y, yb, b, kValid[b], "output");
      expect_valid_rows_match<kSeq>(dx, dxb, b, kValid[b], "dx");
      for (std::size_t r = kValid[b]; r < kSeq; ++r) {
        for (std::size_t d = 0; d < kDim; ++d) EXPECT_EQ(y(b * kSeq + r, d), 0.0f);
      }
    }
    expect_tensors_near(batched.query_proj().weight_grad(),
                        single.query_proj().weight_grad(), kTol, "dWq");
    expect_tensors_near(batched.value_proj().weight_grad(),
                        single.value_proj().weight_grad(), kTol, "dWv");
    expect_tensors_near(batched.output_proj().bias_grad(),
                        single.output_proj().bias_grad(), kTol, "dbo");
  }
}

TEST(BatchedTransformer, MatchesOneExampleAtATime) {
  std::mt19937 rng(6);
  zf::Transformer<float, kSeq, kDim, kFf, zf::Activation::kGelu, kB> batched(2, 2);
  std::mt19937 init(12);
  batched.randomize(init);
  zf::Transformer<float, kSeq, kDim, kFf> single(2, 2);
  init.seed(12);
  single.randomize(init);

  zf::Tensor<float, kB * kSeq, kDim> x;
  zf::Tensor<float, kB * kSeq, kDim> g;
  fill(x, rng);
  fill(g, rng);
  zero_padding<kSeq>(g, kValid);
  for (std::size_t b = 0; b < kB; ++b) batched.set_valid_length(b, kValid[b]);
  const auto y = batched.forward(x);
  const auto dx = batched.backward(g);

  for (std::size_t b = 0; b < kB; ++b) {
    single.set_valid_length(kValid[b]);
    const auto yb = single.forward(slice<kSeq>(x, b));
    const auto dxb = single.backward(slice<kSeq>(g, b));
    expect_valid_rows_match<kSeq>(y, yb, b, kValid[b], "output");
    expect_valid_rows_match<kSeq>(dx, dxb, b, kValid[b], "dx");
  }
  expect_tensors_near(batched.block(1).feed_forward().input_layer().weight_grad(),
                      single.block(1).feed_forward().input_layer().weight_grad(),
                      kTol, "ffn dW");
  expect_tensors_near(batched.final_norm().gamma_grad(),
                      single.final_norm().gamma_grad(), kTol, "final dgamma");
}

TEST(BatchedDecoder, MatchesOneExampleAtATime) {
  std::mt19937 rng(7);
  zf::TransformerDecoder<float, kSeq, kMem, kDim, kFf, zf::Activation::kGelu, kB> batched(2, 2);
  std::mt19937 init(13);
  batched.randomize(init);
  zf::TransformerDecoder<float, kSeq, kMem, kDim, kFf> single(2, 2);
  init.seed(13);
  single.randomize(init);

  zf::Tensor<float, kB * kSeq, kDim> x;
  zf::Tensor<float, kB * kMem, kDim> memory;
  zf::Tensor<float, kB * kSeq, kDim> g;
  fill(x, rng);
  fill(memory, rng);
  fill(g, rng);
  zero_padding<kSeq>(g, kValid);
  for (std::size_t b = 0; b < kB; ++b) {
    batched.set_valid_length(b, kValid[b]);
    batched.set_memory_length(b, kMemValid[b]);
  }
  const auto y = batched.forward(x, memory);
  zf::Tensor<float, kB * kMem, kDim> d_memory;
  const auto dx = batched.backward(g, &d_memory);

  for (std::size_t b = 0; b < kB; ++b) {
    single.set_valid_length(kValid[b]);
    single.set_memory_length(kMemValid[b]);
    const auto yb = single.forward(slice<kSeq>(x, b), slice<kMem>(memory, b));
    zf::Tensor<float, kMem, kDim> d_memory_b;
    const auto dxb = single.backward(slice<kSeq>(g, b), &d_memory_b);
    expect_valid_rows_match<kSeq>(y, yb, b, kValid[b], "output");
    expect_valid_rows_match<kSeq>(dx, dxb, b, kValid[b], "dx");
    expect_valid_rows_match<kMem>(d_memory, d_memory_b, b, kMemValid[b], "d_memory");
    for (std::size_t r = kMemValid[b]; r < kMem; ++r) {
      for (std::size_t d = 0; d < kDim; ++d) EXPECT_EQ(d_memory(b * kMem + r, d), 0.0f);
    }
  }
  auto& cb = batched.block(0).cross_attention();
  auto& cs = single.block(0).cross_attention();
  expect_tensors_near(cb.key_proj().weight_grad(), cs.key_proj().weight_grad(), kTol,
                      "cross dWk");
  expect_tensors_near(cb.query_proj().weight_grad(), cs.query_proj().weight_grad(),
                      kTol, "cross dWq");
  expect_tensors_near(batched.output_norm().beta_grad(),
                      single.output_norm().beta_grad(), kTol, "output dbeta");
}

TEST(BatchedDecoder, OneExampleCannotInfluenceAnother) {
  std::mt19937 rng(8);
  zf::TransformerDecoder<float, kSeq, kMem, kDim, kFf, zf::Activation::kGelu, kB> decoder(2, 2);
  decoder.randomize(rng);
  zf::Tensor<float, kB * kSeq, kDim> x;
  zf::Tensor<float, kB * kMem, kDim> memory;
  fill(x, rng);
  fill(memory, rng);
  for (std::size_t b = 0; b < kB; ++b) {
    decoder.set_valid_length(b, kValid[b]);
    decoder.set_memory_length(b, kMemValid[b]);
  }
  const auto before = decoder.forward(x, memory);
  // Change everything about example 1, and the padding of example 2.
  for (std::size_t r = 0; r < kSeq; ++r) {
    for (std::size_t d = 0; d < kDim; ++d) x(1 * kSeq + r, d) += 3.0f;
  }
  for (std::size_t r = 0; r < kMem; ++r) {
    for (std::size_t d = 0; d < kDim; ++d) memory(1 * kMem + r, d) -= 2.0f;
  }
  for (std::size_t r = kMemValid[2]; r < kMem; ++r) memory(2 * kMem + r, 0) = 99.0f;
  const auto after = decoder.forward(x, memory);
  for (std::size_t b : {std::size_t{0}, std::size_t{2}}) {
    for (std::size_t r = 0; r < kValid[b]; ++r) {
      for (std::size_t d = 0; d < kDim; ++d) {
        EXPECT_EQ(before(b * kSeq + r, d), after(b * kSeq + r, d)) << b << "," << r;
      }
    }
  }
}

TEST(BatchedDecoder, GradientsMatchFiniteDifferences) {
  constexpr std::size_t kB2 = 2;
  zf::TransformerDecoder<float, 3, 4, kDim, kFf, zf::Activation::kGelu, kB2> decoder(2, 2);
  std::mt19937 rng(9);
  decoder.randomize(rng);
  decoder.set_valid_length(1, 2);
  decoder.set_memory_length(0, 3);
  zf::Tensor<float, kB2 * 3, kDim> x;
  zf::Tensor<float, kB2 * 4, kDim> memory;
  zf::Tensor<float, kB2 * 3, kDim> weights;  // a weighted sum, padding weighted 0
  fill(x, rng);
  fill(memory, rng);
  fill(weights, rng);
  for (std::size_t d = 0; d < kDim; ++d) weights(1 * 3 + 2, d) = 0.0f;

  const auto loss = [&]() {
    const auto out = decoder.forward(x, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) sum += out[i] * weights[i];
    return sum;
  };
  loss();
  zf::Tensor<float, kB2 * 4, kDim> d_memory;
  const auto dx = decoder.backward(weights, &d_memory);
  zf_test::expect_grads_match(zf_test::slots_of(x), zf_test::values_of(dx), loss,
                              "batched dx");
  zf_test::expect_grads_match(zf_test::slots_of(memory), zf_test::values_of(d_memory),
                              loss, "batched d_memory");
  auto& wq = decoder.block(1).cross_attention().query_proj();
  zf_test::expect_grads_match(zf_test::slots_of(wq.weight()),
                              zf_test::values_of(wq.weight_grad()), loss,
                              "batched cross dWq");
}

// --- determinism -----------------------------------------------------------

TEST(Determinism, SameSeedTrainsToIdenticalWeights) {
  const auto train = [] {
    zf::TransformerDecoder<float, kSeq, kMem, kDim, kFf, zf::Activation::kGelu, kB> decoder(2, 2);
    std::mt19937 rng(21);
    decoder.randomize(rng);
    zf::Tensor<float, kB * kSeq, kDim> x;
    zf::Tensor<float, kB * kMem, kDim> memory;
    for (int step = 0; step < 5; ++step) {
      fill(x, rng);
      fill(memory, rng);
      decoder.zero_grad();
      const auto y = decoder.forward(x, memory);
      decoder.backward(y, nullptr);  // d(0.5 |y|^2)
      decoder.apply_sgd(0.05f);
    }
    return decoder.block(1).feed_forward().output_layer().weight();
  };
  EXPECT_EQ(train(), train());
}

}  // namespace

namespace {

// --- fast tanh -------------------------------------------------------------

TEST(FastTanh, MatchesTheLibraryAcrossTheRange) {
  double worst = 0.0;
  for (int i = -200000; i <= 200000; ++i) {
    const float u = static_cast<float>(i) * 1e-4f;  // [-20, 20]
    const double want = std::tanh(static_cast<double>(u));
    const double got = zf::fast_tanh(u);
    worst = std::max(worst, std::abs(got - want));
  }
  EXPECT_LT(worst, 3e-7);
  EXPECT_EQ(zf::fast_tanh(0.0f), 0.0f);
  EXPECT_EQ(zf::fast_tanh(100.0f), 1.0f);
  EXPECT_EQ(zf::fast_tanh(-100.0f), -1.0f);
  EXPECT_TRUE(std::isnan(zf::fast_tanh(std::numeric_limits<float>::quiet_NaN())));
}

TEST(FastTanh, StoredGeluDerivativeMatchesTheRecomputedOne) {
  for (int i = -800; i <= 800; ++i) {
    const float x = static_cast<float>(i) * 0.01f;
    float grad = 0.0f;
    const float y = zf::activation_apply_with_grad(zf::Activation::kGelu, x, &grad);
    EXPECT_EQ(y, zf::activation_apply(zf::Activation::kGelu, x));
    EXPECT_EQ(grad, zf::activation_grad(zf::Activation::kGelu, x));
  }
}

}  // namespace

namespace {

// --- activation as a template parameter ------------------------------------

TEST(ReluDecoder, GradientsMatchFiniteDifferences) {
  zf::TransformerDecoder<float, 3, 4, 8, 16, zf::Activation::kRelu, 2> decoder(2, 2);
  static_assert(decltype(decoder)::Block::FF::activation == zf::Activation::kRelu);
  std::mt19937 rng(31);
  decoder.randomize(rng);
  zf::Tensor<float, 6, 8> x;
  zf::Tensor<float, 8, 8> memory;
  zf::Tensor<float, 6, 8> weights;
  fill(x, rng);
  fill(memory, rng);
  fill(weights, rng);
  const auto loss = [&]() {
    const auto out = decoder.forward(x, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) sum += out[i] * weights[i];
    return sum;
  };
  loss();
  zf::Tensor<float, 8, 8> d_memory;
  const auto dx = decoder.backward(weights, &d_memory);
  // A smaller step: ReLU's kink makes wide central differences unreliable.
  zf_test::expect_grads_match(zf_test::slots_of(x), zf_test::values_of(dx), loss,
                              "relu dx", 2e-2, 2e-4f);
  zf_test::expect_grads_match(zf_test::slots_of(memory), zf_test::values_of(d_memory),
                              loss, "relu d_memory", 2e-2, 2e-4f);
}

TEST(ReluDecoder, DiffersFromGeluWithTheSameWeights) {
  zf::TransformerDecoder<float, 3, 4, 8, 16, zf::Activation::kRelu> relu(1, 2);
  zf::TransformerDecoder<float, 3, 4, 8, 16, zf::Activation::kGelu> gelu(1, 2);
  std::mt19937 a(5), b(5);
  relu.randomize(a);
  gelu.randomize(b);
  zf::Tensor<float, 3, 8> x;
  zf::Tensor<float, 4, 8> memory;
  std::mt19937 rng(6);
  fill(x, rng);
  fill(memory, rng);
  EXPECT_NE(relu.forward(x, memory), gelu.forward(x, memory));
}

}  // namespace
