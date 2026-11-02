// bf16 as the element type. The conversion itself is checked bit for bit;
// layers and models are checked against the same model in float, built from
// the same seed, within what three significant digits allow; and one model
// is trained end to end in bf16.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include "add.h"
#include "cross_attention.h"
#include "dot.h"
#include "layers.h"
#include "loss.h"
#include "tensor.h"
#include "transformer.h"

namespace {

using zf::bf16;

std::uint16_t bits_of(float f) { return bf16(f).bits(); }

float from_bits(std::uint32_t u) { return std::bit_cast<float>(u); }

// --- the type ----------------------------------------------------------------

TEST(Bf16, WideningIsExactAndKeepsTheTopHalf) {
  for (float f : {0.0f, -0.0f, 1.0f, -2.5f, 0.15625f, 65280.0f}) {
    EXPECT_EQ(static_cast<float>(bf16(f)), f) << f;  // all exactly representable
  }
  EXPECT_EQ(bits_of(1.0f), 0x3f80);
  EXPECT_EQ(bits_of(-2.0f), 0xc000);
}

TEST(Bf16, RoundsToNearestEven) {
  // 1 + 2^-8 is exactly halfway between 1 and the next bf16 (1 + 2^-7):
  // ties go to the even mantissa, which is 1.
  EXPECT_EQ(bits_of(from_bits(0x3f808000)), 0x3f80);
  // Halfway above an odd mantissa rounds up to the even one.
  EXPECT_EQ(bits_of(from_bits(0x3f818000)), 0x3f82);
  // Anything past halfway rounds up, anything short of it down.
  EXPECT_EQ(bits_of(from_bits(0x3f808001)), 0x3f81);
  EXPECT_EQ(bits_of(from_bits(0x3f807fff)), 0x3f80);
}

TEST(Bf16, SpecialValues) {
  const float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(static_cast<float>(bf16(inf)), inf);
  EXPECT_EQ(static_cast<float>(bf16(-inf)), -inf);
  // A NaN whose payload is all in the low half must not round into infinity.
  EXPECT_TRUE(std::isnan(static_cast<float>(bf16(from_bits(0x7f800001)))));
  EXPECT_TRUE(std::isnan(static_cast<float>(bf16(std::nanf("")))));
  // The largest float rounds past the largest bf16, to infinity.
  EXPECT_EQ(static_cast<float>(bf16(std::numeric_limits<float>::max())), inf);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<bf16>::max()),
            from_bits(0x7f7f0000));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<bf16>::epsilon()), 1.0f / 128);
}

TEST(Bf16, TensorsConvertAndUseFloatSums) {
  zf::TensorF<4> f{1.0f, 0.1f, -3.0f, 1e-3f};
  const auto b = zf::tensor_cast<bf16>(f);
  const auto back = zf::tensor_cast<float>(b);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(back[i], f[i], std::abs(f[i]) / 256) << i;
  }
  // dot accumulates in float, so it is not limited to bf16 precision.
  static_assert(std::is_same_v<decltype(zf::dot(b, b)), float>);
  zf::TensorB<4> ones(bf16(1.0f));
  zf::TensorB<4> tiny(bf16(1.0f / 1024));
  EXPECT_FLOAT_EQ(zf::dot(ones, tiny) + 256.0f, 256.0f + 4.0f / 1024);
  EXPECT_TRUE(zf::all_finite(b));
}

// --- layers against float ------------------------------------------------------

template <typename A, typename B>
double max_rel_error(const A& a, const B& b) {
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    scale = std::max(scale, std::abs(static_cast<double>(b[i])));
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(a[i]) -
                                     static_cast<double>(b[i])));
  }
  return worst / std::max(scale, 1e-30);
}

template <typename T, std::size_t... Dims>
void fill(zf::Tensor<T, Dims...>& t, std::mt19937& rng, float stddev = 0.7f) {
  std::normal_distribution<float> dist(0.0f, stddev);
  for (T& v : t) v = static_cast<T>(dist(rng));
}

// bf16 keeps 8 significant bits, so one rounding is 2^-9 relative. A layer
// rounds its inputs, weights and output; a stack compounds that.
constexpr double kLayerTol = 1e-2;
constexpr double kModelTol = 4e-2;

TEST(Bf16Linear, MatchesFloatWithinRounding) {
  constexpr std::size_t kRows = 11, kIn = 32, kOut = 24;
  std::mt19937 init(1);
  zf::Linear<float, kIn, kOut> lf;
  lf.randomize(init);
  init.seed(1);
  zf::Linear<bf16, kIn, kOut> lb;
  lb.randomize(init);

  std::mt19937 rng(2);
  zf::TensorF<kRows, kIn> x;
  zf::TensorF<kRows, kOut> g;
  fill(x, rng);
  fill(g, rng);
  const auto yf = lf.forward<kRows>(x);
  const auto dxf = lf.backward<kRows>(x, g);
  const auto yb = lb.forward<kRows>(zf::tensor_cast<bf16>(x));
  const auto dxb = lb.backward<kRows>(zf::tensor_cast<bf16>(x), zf::tensor_cast<bf16>(g));
  EXPECT_LT(max_rel_error(yb, yf), kLayerTol);
  EXPECT_LT(max_rel_error(dxb, dxf), kLayerTol);
  // Parameter gradients are float even for bf16 weights.
  static_assert(std::is_same_v<std::decay_t<decltype(lb.weight_grad())>,
                               zf::TensorF<kOut, kIn>>);
  EXPECT_LT(max_rel_error(lb.weight_grad(), lf.weight_grad()), kLayerTol);
  EXPECT_LT(max_rel_error(lb.bias_grad(), lf.bias_grad()), kLayerTol);
}

// A stack of 8+ bf16 rows widens W and the rows to float once and runs the
// float kernel. Widening is exact and the order of the sums is the same, so it
// must match one row at a time bit for bit -- except from blas::gemm_rows on,
// where SGEMM's own order is held to a tolerance.
template <std::size_t kRows>
void check_widened_stack(bool exact) {
  constexpr std::size_t kIn = 12, kOut = 10;  // leftover lanes and outputs
  std::mt19937 rng(7);
  zf::Linear<bf16, kIn, kOut> stacked;
  stacked.randomize(rng);
  zf::Linear<bf16, kIn, kOut> single = stacked;
  zf::TensorB<kRows, kIn> x;
  zf::TensorB<kRows, kOut> g;
  fill(x, rng);
  fill(g, rng);
  const auto y = stacked.template forward<kRows>(x);
  const auto dx = stacked.template backward<kRows>(x, g);
  zf::TensorB<kRows, kOut> y1;
  zf::TensorB<kRows, kIn> dx1;
  for (std::size_t r = 0; r < kRows; ++r) {
    single.forward_into(x.data() + r * kIn, y1.data() + r * kOut, 1);
    single.backward_into(x.data() + r * kIn, g.data() + r * kOut, dx1.data() + r * kIn, 1);
  }
  if (exact) {
    EXPECT_EQ(y, y1);
    EXPECT_EQ(dx, dx1);
    EXPECT_EQ(stacked.weight_grad(), single.weight_grad());
  } else {
    EXPECT_LT(max_rel_error(y, y1), kLayerTol);
    EXPECT_LT(max_rel_error(dx, dx1), kLayerTol);
    EXPECT_LT(max_rel_error(stacked.weight_grad(), single.weight_grad()), 1e-5);
  }
}

TEST(Bf16Linear, WidenedStackMatchesOneRowAtATime) {
  check_widened_stack<11>(/*exact=*/true);
  check_widened_stack<zf::blas::gemm_rows(12, 10) + 3>(/*exact=*/!zf::blas::kEnabled);
}

TEST(Bf16Softmax, RowSumsToOneWithinRounding) {
  bf16 row[7];
  for (int i = 0; i < 7; ++i) row[i] = bf16(0.3f * static_cast<float>(i) - 1.0f);
  zf::softmax_inplace(row, 7);
  float sum = 0.0f;
  for (bf16 v : row) sum += v;
  EXPECT_NEAR(sum, 1.0f, 7.0 / 256);
}

TEST(Bf16Transformer, MatchesFloatWithinRounding) {
  constexpr std::size_t kB = 3, kSeq = 5, kDim = 16, kFf = 32;
  constexpr std::size_t kValid[kB] = {5, 3, 1};
  std::mt19937 init(3);
  zf::Transformer<float, kSeq, kDim, kFf, zf::Activation::kGelu, kB> mf(2, 2);
  mf.randomize(init);
  init.seed(3);
  zf::Transformer<bf16, kSeq, kDim, kFf, zf::Activation::kGelu, kB> mb(2, 2);
  mb.randomize(init);
  for (std::size_t b = 0; b < kB; ++b) {
    mf.set_valid_length(b, kValid[b]);
    mb.set_valid_length(b, kValid[b]);
  }

  std::mt19937 rng(4);
  zf::TensorF<kB * kSeq, kDim> x;
  zf::TensorF<kB * kSeq, kDim> g;
  fill(x, rng);
  fill(g, rng);
  for (std::size_t b = 0; b < kB; ++b) {
    for (std::size_t r = kValid[b]; r < kSeq; ++r) {
      for (std::size_t d = 0; d < kDim; ++d) g(b * kSeq + r, d) = 0.0f;
    }
  }
  const auto yf = mf.forward(x);
  const auto dxf = mf.backward(g);
  const auto yb = mb.forward(zf::tensor_cast<bf16>(x));
  const auto dxb = mb.backward(zf::tensor_cast<bf16>(g));
  EXPECT_LT(max_rel_error(yb, yf), kModelTol);
  EXPECT_LT(max_rel_error(dxb, dxf), kModelTol);
  EXPECT_LT(max_rel_error(mb.block(0).attention().query_proj().weight_grad(),
                          mf.block(0).attention().query_proj().weight_grad()),
            kModelTol);
}

TEST(Bf16Decoder, MatchesFloatWithinRounding) {
  constexpr std::size_t kQ = 3, kMem = 6, kDim = 16, kFf = 32;
  std::mt19937 init(5);
  zf::TransformerDecoder<float, kQ, kMem, kDim, kFf> df(2, 4);
  df.randomize(init);
  init.seed(5);
  zf::TransformerDecoder<bf16, kQ, kMem, kDim, kFf> db(2, 4);
  db.randomize(init);
  df.set_memory_length(4);
  db.set_memory_length(4);

  std::mt19937 rng(6);
  zf::TensorF<kQ, kDim> x;
  zf::TensorF<kMem, kDim> memory;
  zf::TensorF<kQ, kDim> g;
  fill(x, rng);
  fill(memory, rng);
  fill(g, rng);
  const auto yf = df.forward(x, memory);
  zf::TensorF<kMem, kDim> dmf;
  const auto dxf = df.backward(g, &dmf);
  const auto yb = db.forward(zf::tensor_cast<bf16>(x), zf::tensor_cast<bf16>(memory));
  zf::TensorB<kMem, kDim> dmb;
  const auto dxb = db.backward(zf::tensor_cast<bf16>(g), &dmb);
  EXPECT_LT(max_rel_error(yb, yf), kModelTol);
  EXPECT_LT(max_rel_error(dxb, dxf), kModelTol);
  EXPECT_LT(max_rel_error(dmb, dmf), kModelTol);
}

// --- training ------------------------------------------------------------------

// The README's pointing task -- four memory tokens, one query naming a
// symbol, answer the slot that holds it -- written once for any element type.
template <typename T>
double train_pointer(int steps) {
  constexpr std::size_t kVocab = 8, kMem = 4, kQuery = 1, kDim = 16, kFf = 32;
  std::mt19937_64 rng(1);
  zf::Embedding<T, kVocab, kDim> symbol;
  zf::Embedding<T, kMem, kDim> position;
  zf::Embedding<T, kVocab, kDim> ask;
  zf::TransformerDecoder<T, kQuery, kMem, kDim, kFf> decoder(2, 2);
  zf::Linear<T, kDim, kMem> head;
  zf::SoftmaxCrossEntropyLoss<T, kMem> loss;
  symbol.randomize(rng, static_cast<T>(0.5));
  position.randomize(rng, static_cast<T>(0.5));
  ask.randomize(rng, static_cast<T>(0.5));
  decoder.randomize(rng);
  head.randomize(rng);

  std::uniform_int_distribution<std::size_t> pick_symbol(0, kVocab - 1);
  std::uniform_int_distribution<std::size_t> pick_slot(0, kMem - 1);
  const int window = 1000;
  std::size_t correct = 0;
  for (int step = 1; step <= steps; ++step) {
    std::size_t ids[kMem];
    for (std::size_t& id : ids) id = pick_symbol(rng);
    const std::size_t answer = pick_slot(rng);

    zf::Tensor<T, kMem, kDim> memory;
    for (std::size_t m = 0; m < kMem; ++m) {
      const auto s = symbol.forward(ids[m]);
      const auto p = position.forward(m);
      for (std::size_t d = 0; d < kDim; ++d) memory(m, d) = static_cast<T>(s(d) + p(d));
    }
    zf::Tensor<T, kQuery, kDim> query;
    const auto q = ask.forward(ids[answer]);
    for (std::size_t d = 0; d < kDim; ++d) query(0, d) = q(d);

    const auto h = decoder.forward(query, memory);
    zf::Tensor<T, kDim> h0;
    for (std::size_t d = 0; d < kDim; ++d) h0(d) = h(0, d);
    loss.forward(head.forward(h0), answer);
    if (step > steps - window) correct += ids[loss.argmax()] == ids[answer];

    symbol.zero_grad(); position.zero_grad(); ask.zero_grad();
    decoder.zero_grad(); head.zero_grad();
    const auto d_h0 = head.backward(h0, loss.backward());
    zf::Tensor<T, kQuery, kDim> d_h;
    for (std::size_t d = 0; d < kDim; ++d) d_h(0, d) = d_h0(d);
    zf::Tensor<T, kMem, kDim> d_memory;
    const auto d_query = decoder.backward(d_h, &d_memory);
    zf::Tensor<T, kDim> row;
    for (std::size_t d = 0; d < kDim; ++d) row(d) = d_query(0, d);
    ask.backward(ids[answer], row);
    for (std::size_t m = 0; m < kMem; ++m) {
      for (std::size_t d = 0; d < kDim; ++d) row(d) = d_memory(m, d);
      symbol.backward(ids[m], row);
      position.backward(m, row);
    }
    const float lr = 0.02f;
    symbol.apply_sgd(lr); position.apply_sgd(lr); ask.apply_sgd(lr);
    decoder.apply_sgd(lr); head.apply_sgd(lr);
  }
  return static_cast<double>(correct) / window;
}

TEST(Bf16Training, LearnsToPoint) {
  const double acc_float = train_pointer<float>(8000);
  const double acc_bf16 = train_pointer<bf16>(8000);
  std::printf("  accuracy after 8000 steps: float %.3f, bf16 %.3f\n", acc_float,
              acc_bf16);
  EXPECT_GT(acc_bf16, 0.9);
}

}  // namespace
