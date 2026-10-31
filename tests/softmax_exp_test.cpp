#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "layers.h"
#include "tensor.h"

namespace {

// The softmax uses its own exp rather than the library's, for speed. That is
// only acceptable while it agrees with the library's to the accuracy of the
// type, so this is the check that licenses it.
TEST(SoftmaxExp, MatchesTheLibraryAcrossTheWholeRange) {
  double worst_relative = 0.0;
  float worst_at = 0.0f;
  // Softmax subtracts the row maximum, so its arguments are never positive.
  // -87.3 is where 2^k stops being representable in float.
  for (float x = -90.0f; x <= 0.0f; x += 0.0005f) {
    const double want = std::exp(static_cast<double>(x));
    const double got = static_cast<double>(zf::softmax_exp(x));
    if (want < std::numeric_limits<float>::min()) {
      // Below the smallest normal float, zero is the right answer.
      EXPECT_LE(got, static_cast<double>(std::numeric_limits<float>::min()));
      continue;
    }
    const double relative = std::abs(got - want) / want;
    if (relative > worst_relative) {
      worst_relative = relative;
      worst_at = x;
    }
  }
  // Measured at 4.3e-7, which is three or four units in the last place of a
  // float (its epsilon is 1.19e-7). The bound is set just above the measured
  // value so that a change which degrades the approximation fails here rather
  // than passing quietly.
  EXPECT_LT(worst_relative, 6e-7) << "worst at x = " << worst_at;
}

TEST(SoftmaxExp, IsExactAtZero) {
  // exp(0) is the one value a softmax always evaluates -- the row maximum
  // maps to it -- so it has to be exactly one, not one within a tolerance.
  EXPECT_EQ(zf::softmax_exp(0.0f), 1.0f);
}

TEST(SoftmaxExp, HoldsAtThePowersOfTwo) {
  // Near k*ln2 the range reduction leaves r close to zero, so these points
  // exercise the exponent assembly rather than the polynomial.
  for (int k = -80; k <= 0; ++k) {
    const float x = static_cast<float>(k) * 0.69314718055994531f;
    EXPECT_NEAR(zf::softmax_exp(x) / std::exp(x), 1.0f, 6e-7f) << "k = " << k;
  }
}

TEST(SoftmaxExp, UnderflowsToZeroRatherThanMisbehaving) {
  EXPECT_FLOAT_EQ(zf::softmax_exp(-1000.0f), 0.0f);
  EXPECT_FLOAT_EQ(zf::softmax_exp(-std::numeric_limits<float>::infinity()),
                  0.0f);
  // A NaN propagates rather than being turned into a number, the same as
  // std::exp: a NaN reaching a softmax is a bug upstream and hiding it here
  // would only move the symptom.
  EXPECT_TRUE(std::isnan(
      zf::softmax_exp(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_GT(zf::softmax_exp(-80.0f), 0.0f);
  // Everything down to the smallest normal float is still produced; only
  // below that does it give zero.
  EXPECT_GT(zf::softmax_exp(-87.3f), 0.0f);
  EXPECT_NEAR(zf::softmax_exp(-87.3f) / std::exp(-87.3f), 1.0f, 6e-7f);
  EXPECT_FLOAT_EQ(zf::softmax_exp(-90.0f), 0.0f);

}

TEST(SoftmaxExp, ASoftmaxStillSumsToOne) {
  std::mt19937 rng(4);
  std::normal_distribution<float> dist(0.0f, 8.0f);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<float> row(48);
    for (float& v : row) {
      v = dist(rng);
    }
    std::vector<float> reference = row;

    zf::softmax_inplace(row.data(), row.size());

    // Against a plainly written softmax over double.
    double max_v = reference[0];
    for (float v : reference) {
      max_v = std::max(max_v, static_cast<double>(v));
    }
    double sum = 0.0;
    std::vector<double> want(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
      want[i] = std::exp(static_cast<double>(reference[i]) - max_v);
      sum += want[i];
    }

    double total = 0.0;
    for (std::size_t i = 0; i < row.size(); ++i) {
      total += row[i];
      EXPECT_NEAR(row[i], want[i] / sum, 1e-6) << "element " << i;
      EXPECT_GE(row[i], 0.0f);
    }
    EXPECT_NEAR(total, 1.0, 1e-5);
  }
}

TEST(SoftmaxExp, HandlesARowOfEqualAndOfExtremeValues) {
  // All equal: a uniform distribution.
  std::vector<float> flat(8, 3.5f);
  zf::softmax_inplace(flat.data(), flat.size());
  for (float v : flat) {
    EXPECT_NEAR(v, 0.125f, 1e-6f);
  }

  // One value far above the rest: it takes essentially all the mass, and the
  // others underflow to zero without producing a NaN.
  std::vector<float> spiked = {100.0f, -100.0f, -100.0f, -100.0f};
  zf::softmax_inplace(spiked.data(), spiked.size());
  EXPECT_NEAR(spiked[0], 1.0f, 1e-6f);
  for (std::size_t i = 1; i < spiked.size(); ++i) {
    EXPECT_FLOAT_EQ(spiked[i], 0.0f);
  }

  // A single element is always 1.
  std::vector<float> one = {-12.0f};
  zf::softmax_inplace(one.data(), one.size());
  EXPECT_FLOAT_EQ(one[0], 1.0f);
}

TEST(SoftmaxExp, DoubleStillUsesTheLibrary) {
  // The approximation is float-only; a double instantiation must keep full
  // double accuracy rather than silently inherit a float-accurate exp.
  std::vector<double> row = {0.5, -0.25, 2.0};
  const std::vector<double> input = row;
  zf::softmax_inplace(row.data(), row.size());
  double sum = 0.0;
  for (double v : std::vector<double>{std::exp(input[0] - 2.0),
                                      std::exp(input[1] - 2.0),
                                      std::exp(input[2] - 2.0)}) {
    sum += v;
  }
  EXPECT_NEAR(row[0], std::exp(input[0] - 2.0) / sum, 1e-15);
  EXPECT_NEAR(row[2], std::exp(input[2] - 2.0) / sum, 1e-15);
}

}  // namespace
