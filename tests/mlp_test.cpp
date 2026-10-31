#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "dot.h"
#include "grad_check.h"
#include "loss.h"
#include "mlp.h"

namespace {

using zf_test::expect_grads_match;
using zf_test::randomize_tensor;
using zf_test::slots_of;
using zf_test::values_of;

TEST(Mlp, DimensionsAreCompileTime) {
  using M = zf::Mlp<float, 4, 8, 2>;
  static_assert(M::in_dim == 4);
  static_assert(M::hidden_dim == 8);
  static_assert(M::out_dim == 2);
  // 4*8+8 + 8*2+2 = 40 + 18
  static_assert(M::parameter_count == 58);
}

TEST(Mlp, ForwardShape) {
  auto gen = zf_test::rng();
  zf::Mlp<float, 4, 8, 2> mlp;
  mlp.randomize(gen);
  zf::TensorF<4> x;
  randomize_tensor(x, gen);
  const auto y = mlp.forward(x);
  static_assert(std::is_same_v<std::decay_t<decltype(y)>, zf::TensorF<2>>);
  EXPECT_EQ(y.size(), 2u);
}

TEST(Mlp, ReluZerosNegativeHiddenUnits) {
  zf::Mlp<float, 1, 2, 1> mlp(zf::Activation::kRelu);
  mlp.hidden_layer().weight() = zf::TensorF<2, 1>{1.0f, -1.0f};
  mlp.output_layer().weight() = zf::TensorF<1, 2>{1.0f, 1.0f};
  // x=1 -> hidden pre = (1, -1) -> relu = (1, 0) -> out = 1
  EXPECT_FLOAT_EQ(mlp.forward(zf::TensorF<1>{1.0f})(0), 1.0f);
  // x=-1 -> hidden pre = (-1, 1) -> relu = (0, 1) -> out = 1
  EXPECT_FLOAT_EQ(mlp.forward(zf::TensorF<1>{-1.0f})(0), 1.0f);
}

template <zf::Activation Act>
void CheckMlpGradients(const char* what) {
  auto gen = zf_test::rng();
  zf::Mlp<float, 3, 5, 2> mlp(Act);
  mlp.randomize(gen);
  zf::TensorF<3> x;
  randomize_tensor(x, gen);
  zf::TensorF<2> w;
  randomize_tensor(w, gen);

  const auto loss = [&] { return zf::dot(mlp.forward(x), w); };

  mlp.zero_grad();
  mlp.forward(x);
  const auto dx = mlp.backward(w);

  expect_grads_match(slots_of(mlp.hidden_layer().weight()),
                     values_of(mlp.hidden_layer().weight_grad()), loss,
                     std::string(what) + " fc1 weight");
  expect_grads_match(slots_of(mlp.hidden_layer().bias()),
                     values_of(mlp.hidden_layer().bias_grad()), loss,
                     std::string(what) + " fc1 bias");
  expect_grads_match(slots_of(mlp.output_layer().weight()),
                     values_of(mlp.output_layer().weight_grad()), loss,
                     std::string(what) + " fc2 weight");
  expect_grads_match(slots_of(x), values_of(dx), loss,
                     std::string(what) + " input");
}

TEST(Mlp, GradientsMatchFiniteDifferencesGelu) {
  // GELU is smooth everywhere, so this exercises the full chain rule.
  CheckMlpGradients<zf::Activation::kGelu>("gelu mlp");
}

TEST(Mlp, GradientsMatchFiniteDifferencesIdentity) {
  CheckMlpGradients<zf::Activation::kIdentity>("identity mlp");
}

TEST(Mlp, BackwardAccumulatesUntilZeroGrad) {
  auto gen = zf_test::rng();
  zf::Mlp<float, 2, 3, 1> mlp;
  mlp.randomize(gen);
  zf::TensorF<2> x{1.0f, 1.0f};
  const zf::TensorF<1> d{1.0f};

  mlp.zero_grad();
  mlp.forward(x);
  mlp.backward(d);
  const float once = mlp.output_layer().bias_grad()(0);

  mlp.forward(x);
  mlp.backward(d);
  EXPECT_FLOAT_EQ(mlp.output_layer().bias_grad()(0), 2.0f * once);

  mlp.zero_grad();
  EXPECT_FLOAT_EQ(mlp.output_layer().bias_grad()(0), 0.0f);
}

TEST(Mlp, SgdReducesMseOnAFixedTarget) {
  auto gen = zf_test::rng();
  zf::Mlp<float, 4, 16, 3> mlp(zf::Activation::kRelu);
  mlp.randomize(gen);
  zf::MseLoss<float, 3> loss;

  const zf::TensorF<4> x{0.5f, -0.25f, 1.0f, 0.75f};
  const zf::TensorF<3> target{1.0f, -1.0f, 0.5f};

  const float before = loss.forward(mlp.forward(x), target);
  for (int step = 0; step < 200; ++step) {
    mlp.zero_grad();
    const auto pred = mlp.forward(x);
    loss.forward(pred, target);
    mlp.backward(loss.backward());
    mlp.apply_sgd(0.05f);
  }
  const float after = loss.forward(mlp.forward(x), target);

  EXPECT_LT(after, before);
  EXPECT_LT(after, 1e-4f) << "200 steps should essentially fit one sample";
}

}  // namespace
