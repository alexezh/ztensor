#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "dot.h"
#include "grad_check.h"
#include "layers.h"

namespace {

using zf_test::expect_grads_match;
using zf_test::randomize_tensor;
using zf_test::slots_of;
using zf_test::values_of;

TEST(Activation, Relu) {
  EXPECT_FLOAT_EQ(zf::activation_apply(zf::Activation::kRelu, 2.0f), 2.0f);
  EXPECT_FLOAT_EQ(zf::activation_apply(zf::Activation::kRelu, -2.0f), 0.0f);
  EXPECT_FLOAT_EQ(zf::activation_grad(zf::Activation::kRelu, 2.0f), 1.0f);
  EXPECT_FLOAT_EQ(zf::activation_grad(zf::Activation::kRelu, -2.0f), 0.0f);
}

TEST(Activation, GeluMatchesNumericDerivative) {
  for (float x : {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f}) {
    const float h = 1e-3f;
    const float num =
        (zf::activation_apply(zf::Activation::kGelu, x + h) -
         zf::activation_apply(zf::Activation::kGelu, x - h)) /
        (2.0f * h);
    EXPECT_NEAR(zf::activation_grad(zf::Activation::kGelu, x), num, 1e-3f)
        << "at x=" << x;
  }
  // GELU is ~0 well below zero and ~identity well above.
  EXPECT_NEAR(zf::activation_apply(zf::Activation::kGelu, -6.0f), 0.0f, 1e-4f);
  EXPECT_NEAR(zf::activation_apply(zf::Activation::kGelu, 6.0f), 6.0f, 1e-4f);
}

TEST(Softmax, SumsToOneAndIsShiftInvariant) {
  zf::TensorF<4> a{1.0f, 2.0f, 3.0f, 4.0f};
  zf::TensorF<4> b{101.0f, 102.0f, 103.0f, 104.0f};
  zf::softmax_inplace(a.data(), 4);
  zf::softmax_inplace(b.data(), 4);
  float sum = 0.0f;
  for (std::size_t i = 0; i < 4; ++i) {
    sum += a(i);
    EXPECT_NEAR(a(i), b(i), 1e-6f);
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(Softmax, HandlesPartialRow) {
  // Only the first 2 entries participate; the rest stay untouched.
  zf::TensorF<4> a{0.0f, 0.0f, 7.0f, 7.0f};
  zf::softmax_inplace(a.data(), 2);
  EXPECT_NEAR(a(0), 0.5f, 1e-6f);
  EXPECT_NEAR(a(1), 0.5f, 1e-6f);
  EXPECT_FLOAT_EQ(a(2), 7.0f);
}

TEST(Linear, ForwardIsWeightTimesInputPlusBias) {
  zf::Linear<float, 2, 3> fc;
  fc.weight() = zf::TensorF<3, 2>{1, 2, 3, 4, 5, 6};
  fc.bias() = zf::TensorF<3>{10, 20, 30};
  const zf::TensorF<2> x{1, 1};
  const auto y = fc.forward(x);
  EXPECT_FLOAT_EQ(y(0), 13.0f);
  EXPECT_FLOAT_EQ(y(1), 27.0f);
  EXPECT_FLOAT_EQ(y(2), 41.0f);
}

TEST(Linear, SequenceForwardMatchesPerRowForward) {
  auto gen = zf_test::rng();
  zf::Linear<float, 3, 2> fc;
  fc.randomize(gen);
  zf::TensorF<4, 3> seq;
  randomize_tensor(seq, gen);

  const auto out = fc.forward<4>(seq);
  for (std::size_t r = 0; r < 4; ++r) {
    zf::TensorF<3> row;
    for (std::size_t i = 0; i < 3; ++i) {
      row(i) = seq(r, i);
    }
    const auto expected = fc.forward(row);
    for (std::size_t o = 0; o < 2; ++o) {
      EXPECT_NEAR(out(r, o), expected(o), 1e-6f);
    }
  }
}

TEST(Linear, GradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::Linear<float, 3, 2> fc;
  fc.randomize(gen);
  zf::TensorF<3> x;
  randomize_tensor(x, gen);
  // scalar objective: sum(w . y) for a fixed random w
  zf::TensorF<2> w;
  randomize_tensor(w, gen);

  const auto loss = [&] {
    const auto y = fc.forward(x);
    return w(0) * y(0) + w(1) * y(1);
  };

  fc.zero_grad();
  fc.forward(x);
  const auto dx = fc.backward(x, w);

  expect_grads_match(slots_of(fc.weight()), values_of(fc.weight_grad()), loss,
                     "linear weight");
  expect_grads_match(slots_of(fc.bias()), values_of(fc.bias_grad()), loss,
                     "linear bias");
  expect_grads_match(slots_of(x), values_of(dx), loss, "linear input");
}

TEST(Linear, SequenceGradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::Linear<float, 3, 2> fc;
  fc.randomize(gen);
  zf::TensorF<4, 3> x;
  randomize_tensor(x, gen);
  zf::TensorF<4, 2> w;
  randomize_tensor(w, gen);

  const auto loss = [&] {
    const auto y = fc.forward<4>(x);
    return zf::dot(y, w);
  };

  fc.zero_grad();
  const auto dx = fc.backward<4>(x, w);

  expect_grads_match(slots_of(fc.weight()), values_of(fc.weight_grad()), loss,
                     "linear seq weight");
  expect_grads_match(slots_of(fc.bias()), values_of(fc.bias_grad()), loss,
                     "linear seq bias");
  expect_grads_match(slots_of(x), values_of(dx), loss, "linear seq input");
}

TEST(Linear, SgdStepMovesAgainstTheGradient) {
  zf::Linear<float, 1, 1> fc;
  fc.weight()(0, 0) = 1.0f;
  const zf::TensorF<1> x{2.0f};
  const zf::TensorF<1> d{1.0f};
  fc.zero_grad();
  fc.backward(x, d);  // dw = 2, db = 1
  EXPECT_FLOAT_EQ(fc.weight_grad()(0, 0), 2.0f);
  fc.apply_sgd(0.1f);
  EXPECT_FLOAT_EQ(fc.weight()(0, 0), 1.0f - 0.2f);
  EXPECT_FLOAT_EQ(fc.bias()(0), -0.1f);
}

TEST(LayerNorm, OutputIsNormalized) {
  zf::LayerNorm<float, 4> ln;
  const zf::TensorF<4> x{1, 2, 3, 4};
  const auto y = ln.forward(x);
  float mean = 0.0f;
  for (std::size_t i = 0; i < 4; ++i) {
    mean += y(i);
  }
  mean /= 4.0f;
  float var = 0.0f;
  for (std::size_t i = 0; i < 4; ++i) {
    var += (y(i) - mean) * (y(i) - mean);
  }
  var /= 4.0f;
  EXPECT_NEAR(mean, 0.0f, 1e-5f);
  EXPECT_NEAR(var, 1.0f, 1e-4f);
}

TEST(LayerNorm, GradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::LayerNorm<float, 4> ln;
  randomize_tensor(ln.gamma(), gen, 0.5);
  randomize_tensor(ln.beta(), gen, 0.5);
  for (std::size_t i = 0; i < 4; ++i) {
    ln.gamma()(i) += 1.0f;
  }
  zf::TensorF<4> x;
  randomize_tensor(x, gen);
  zf::TensorF<4> w;
  randomize_tensor(w, gen);

  const auto loss = [&] { return zf::dot(ln.forward(x), w); };

  ln.zero_grad();
  const auto dx = ln.backward(x, w);

  expect_grads_match(slots_of(ln.gamma()), values_of(ln.gamma_grad()), loss,
                     "layernorm gamma");
  expect_grads_match(slots_of(ln.beta()), values_of(ln.beta_grad()), loss,
                     "layernorm beta");
  expect_grads_match(slots_of(x), values_of(dx), loss, "layernorm input");
}

TEST(LayerNorm, SequenceGradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::LayerNorm<float, 4> ln;
  ln.gamma().fill(1.2f);
  randomize_tensor(ln.beta(), gen, 0.3);
  zf::TensorF<3, 4> x;
  randomize_tensor(x, gen);
  zf::TensorF<3, 4> w;
  randomize_tensor(w, gen);

  const auto loss = [&] { return zf::dot(ln.forward<3>(x), w); };

  ln.zero_grad();
  const auto dx = ln.backward<3>(x, w);

  expect_grads_match(slots_of(ln.gamma()), values_of(ln.gamma_grad()), loss,
                     "layernorm seq gamma");
  expect_grads_match(slots_of(x), values_of(dx), loss, "layernorm seq input");
}

TEST(Embedding, LookupAndScatterGrad) {
  zf::Embedding<float, 5, 3> emb;
  emb.weight()(2, 0) = 1.0f;
  emb.weight()(2, 1) = 2.0f;
  emb.weight()(2, 2) = 3.0f;

  const auto e = emb.forward(2);
  EXPECT_FLOAT_EQ(e(1), 2.0f);

  emb.zero_grad();
  emb.backward(2, zf::TensorF<3>{1, 1, 1});
  emb.backward(2, zf::TensorF<3>{1, 1, 1});
  // same id touched twice: gradient accumulates
  EXPECT_FLOAT_EQ(emb.weight_grad()(2, 0), 2.0f);
  EXPECT_FLOAT_EQ(emb.weight_grad()(1, 0), 0.0f);
}

TEST(Embedding, RejectsOutOfVocabId) {
  zf::Embedding<float, 5, 3> emb;
  EXPECT_THROW(emb.forward(5), std::out_of_range);
  EXPECT_THROW(emb.backward(9, zf::TensorF<3>{}), std::out_of_range);
}

}  // namespace
