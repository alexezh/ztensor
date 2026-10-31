#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dot.h"
#include "grad_check.h"
#include "loss.h"

namespace {

using zf_test::expect_grads_match;
using zf_test::randomize_tensor;
using zf_test::slots_of;
using zf_test::values_of;

TEST(MseLoss, ValueIsMeanSquaredError) {
  zf::MseLoss<float, 4> loss;
  const zf::TensorF<4> pred{1, 2, 3, 4};
  const zf::TensorF<4> target{1, 2, 3, 0};
  // (0 + 0 + 0 + 16) / 4
  EXPECT_FLOAT_EQ(loss.forward(pred, target), 4.0f);
}

TEST(MseLoss, ZeroWhenPredictionMatches) {
  zf::MseLoss<float, 3> loss;
  const zf::TensorF<3> p{1, -2, 0.5f};
  EXPECT_FLOAT_EQ(loss.forward(p, p), 0.0f);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(loss.backward()(i), 0.0f);
  }
}

TEST(MseLoss, GradientMatchesFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::MseLoss<float, 5> loss;
  zf::TensorF<5> pred;
  zf::TensorF<5> target;
  randomize_tensor(pred, gen);
  randomize_tensor(target, gen);

  const auto objective = [&] { return loss.forward(pred, target); };
  loss.forward(pred, target);
  const auto grad = values_of(loss.backward());
  expect_grads_match(slots_of(pred), grad, objective, "mse");
}

TEST(MseLoss, WorksOnMatrixShapes) {
  zf::MseLoss<float, 2, 2> loss;
  const zf::TensorF<2, 2> pred{1, 1, 1, 1};
  const zf::TensorF<2, 2> target{0, 0, 0, 0};
  EXPECT_FLOAT_EQ(loss.forward(pred, target), 1.0f);
}

TEST(MseLoss, BackwardBeforeForwardThrows) {
  zf::MseLoss<float, 2> loss;
  EXPECT_THROW(loss.backward(), std::logic_error);
}

TEST(CrossEntropy, UniformLogitsGiveLogN) {
  zf::SoftmaxCrossEntropyLoss<float, 4> loss;
  const zf::TensorF<4> logits{0, 0, 0, 0};
  EXPECT_NEAR(loss.forward(logits, 2), std::log(4.0f), 1e-6f);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(loss.probabilities()(i), 0.25f, 1e-6f);
  }
}

TEST(CrossEntropy, ConfidentCorrectPredictionHasNearZeroLoss) {
  zf::SoftmaxCrossEntropyLoss<float, 3> loss;
  const zf::TensorF<3> logits{20, 0, 0};
  EXPECT_LT(loss.forward(logits, 0), 1e-6f);
  EXPECT_EQ(loss.argmax(), 0u);
}

TEST(CrossEntropy, ConfidentWrongPredictionHasLargeLoss) {
  zf::SoftmaxCrossEntropyLoss<float, 3> loss;
  const zf::TensorF<3> logits{20, 0, 0};
  EXPECT_GT(loss.forward(logits, 1), 10.0f);
}

TEST(CrossEntropy, GradientIsProbabilitiesMinusOneHot) {
  zf::SoftmaxCrossEntropyLoss<float, 4> loss;
  const zf::TensorF<4> logits{0, 0, 0, 0};
  loss.forward(logits, 1);
  EXPECT_NEAR(loss.backward()(0), 0.25f, 1e-6f);
  EXPECT_NEAR(loss.backward()(1), 0.25f - 1.0f, 1e-6f);
  // the gradient of a softmax cross entropy always sums to zero
  float sum = 0.0f;
  for (std::size_t i = 0; i < 4; ++i) {
    sum += loss.backward()(i);
  }
  EXPECT_NEAR(sum, 0.0f, 1e-6f);
}

TEST(CrossEntropy, GradientMatchesFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::SoftmaxCrossEntropyLoss<float, 5> loss;
  zf::TensorF<5> logits;
  randomize_tensor(logits, gen);

  const auto objective = [&] { return loss.forward(logits, 3); };
  loss.forward(logits, 3);
  const auto grad = values_of(loss.backward());
  expect_grads_match(slots_of(logits), grad, objective, "cross entropy");
}

TEST(CrossEntropy, IsShiftInvariantInTheLogits) {
  zf::SoftmaxCrossEntropyLoss<float, 3> a;
  zf::SoftmaxCrossEntropyLoss<float, 3> b;
  const zf::TensorF<3> small{1, 2, 3};
  const zf::TensorF<3> large{1001, 1002, 1003};
  EXPECT_NEAR(a.forward(small, 1), b.forward(large, 1), 1e-5f);
}

TEST(CrossEntropy, RejectsBadLabel) {
  zf::SoftmaxCrossEntropyLoss<float, 3> loss;
  EXPECT_THROW(loss.forward(zf::TensorF<3>{}, 3), std::out_of_range);
  EXPECT_THROW(loss.backward(), std::logic_error);
}

TEST(InfoNce, PerfectMatchWithStrongSeparationIsNearZero) {
  zf::InfoNceLoss<float, 3> loss(0.05f);
  const zf::TensorF<3> q{1, 0, 0};
  const std::vector<zf::TensorF<3>> docs{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  EXPECT_LT(loss.forward(q, docs, 0), 1e-5f);
  EXPECT_EQ(loss.argmax(), 0u);
}

TEST(InfoNce, IdenticalCandidatesGiveLogN) {
  zf::InfoNceLoss<float, 3> loss;
  const zf::TensorF<3> q{1, 1, 1};
  const std::vector<zf::TensorF<3>> docs{{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  EXPECT_NEAR(loss.forward(q, docs, 1), std::log(3.0f), 1e-5f);
}

TEST(InfoNce, ProbabilitiesSumToOne) {
  zf::InfoNceLoss<float, 3> loss(0.5f);
  const zf::TensorF<3> q{0.3f, -0.7f, 1.0f};
  const std::vector<zf::TensorF<3>> docs{
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
  loss.forward(q, docs, 2);
  float sum = 0.0f;
  for (float p : loss.probabilities()) {
    sum += p;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6f);
  EXPECT_EQ(loss.probabilities().size(), 4u);
}

TEST(InfoNce, LowerTemperatureSharpensTheDistribution) {
  const zf::TensorF<3> q{1, 0, 0};
  const std::vector<zf::TensorF<3>> docs{{1, 0, 0}, {0.9f, 0, 0}};

  zf::InfoNceLoss<float, 3> warm(1.0f);
  zf::InfoNceLoss<float, 3> cold(0.05f);
  warm.forward(q, docs, 0);
  cold.forward(q, docs, 0);
  EXPECT_GT(cold.probabilities()[0], warm.probabilities()[0]);
}

TEST(InfoNce, QueryGradientMatchesFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::InfoNceLoss<float, 4> loss(0.25f);
  zf::TensorF<4> q;
  randomize_tensor(q, gen, 0.5);
  std::vector<zf::TensorF<4>> docs(3);
  for (auto& d : docs) {
    randomize_tensor(d, gen, 0.5);
  }

  const auto objective = [&] { return loss.forward(q, docs, 1); };
  loss.forward(q, docs, 1);
  const auto grad = values_of(loss.query_grad());
  expect_grads_match(slots_of(q), grad, objective, "infonce query");
}

TEST(InfoNce, CandidateGradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  zf::InfoNceLoss<float, 4> loss(0.25f);
  zf::TensorF<4> q;
  randomize_tensor(q, gen, 0.5);
  std::vector<zf::TensorF<4>> docs(3);
  for (auto& d : docs) {
    randomize_tensor(d, gen, 0.5);
  }

  const auto objective = [&] { return loss.forward(q, docs, 1); };
  for (std::size_t i = 0; i < docs.size(); ++i) {
    loss.forward(q, docs, 1);
    const auto grad = values_of(loss.candidate_grad(i));
    expect_grads_match(slots_of(docs[i]), grad, objective,
                       "infonce candidate " + std::to_string(i));
  }
}

TEST(InfoNce, RejectsBadArguments) {
  EXPECT_THROW((zf::InfoNceLoss<float, 3>(0.0f)), std::invalid_argument);
  EXPECT_THROW((zf::InfoNceLoss<float, 3>(-1.0f)), std::invalid_argument);

  zf::InfoNceLoss<float, 3> loss;
  const zf::TensorF<3> q{1, 0, 0};
  const std::vector<zf::TensorF<3>> empty;
  const std::vector<zf::TensorF<3>> one{{1, 0, 0}};
  EXPECT_THROW(loss.forward(q, empty, 0), std::invalid_argument);
  EXPECT_THROW(loss.forward(q, one, 1), std::out_of_range);
  EXPECT_THROW(loss.query_grad(), std::logic_error);
}

}  // namespace
