#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "dot.h"
#include "mul.h"
#include "tensor.h"

namespace {

TEST(Tensor, ShapeAndSizeAreCompileTime) {
  using M = zf::TensorF<2, 3>;
  static_assert(M::rank == 2);
  static_assert(M::element_count == 6);
  static_assert(M::shape[0] == 2 && M::shape[1] == 3);
  static_assert(M::strides()[0] == 3 && M::strides()[1] == 1);
  EXPECT_EQ(M::size(), 6u);
  EXPECT_EQ(M::extent(1), 3u);
}

TEST(Tensor, DefaultIsZeroFilled) {
  zf::TensorF<2, 2> t;
  for (float v : t) {
    EXPECT_FLOAT_EQ(v, 0.0f);
  }
}

TEST(Tensor, FillConstructorAndFill) {
  zf::TensorF<2, 2> t(1.5f);
  EXPECT_FLOAT_EQ(t(1, 1), 1.5f);
  t.fill(-2.0f);
  EXPECT_FLOAT_EQ(t(0, 1), -2.0f);
}

TEST(Tensor, RowMajorLayout) {
  zf::TensorF<2, 3> t{1, 2, 3, 4, 5, 6};
  EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t(0, 2), 3.0f);
  EXPECT_FLOAT_EQ(t(1, 0), 4.0f);
  EXPECT_FLOAT_EQ(t(1, 2), 6.0f);
  // Flat access sees the same buffer in row-major order.
  EXPECT_FLOAT_EQ(t[4], 5.0f);
  EXPECT_EQ(t.data()[5], 6.0f);
}

TEST(Tensor, WriteThroughAccessor) {
  zf::TensorF<2, 2> t;
  t(0, 1) = 7.0f;
  EXPECT_FLOAT_EQ(t[1], 7.0f);
}

TEST(Tensor, RankThree) {
  zf::TensorF<2, 3, 4> t;
  EXPECT_EQ(t.size(), 24u);
  t(1, 2, 3) = 9.0f;
  EXPECT_FLOAT_EQ(t[23], 9.0f);
  EXPECT_FLOAT_EQ(t(1, 2, 3), 9.0f);
}

TEST(Tensor, AtIsBoundsChecked) {
  zf::TensorF<2, 2> t{1, 2, 3, 4};
  EXPECT_FLOAT_EQ(t.at(1, 1), 4.0f);
  EXPECT_THROW(t.at(2, 0), std::out_of_range);
  EXPECT_THROW(t.at(0, 2), std::out_of_range);
}

TEST(Tensor, InitializerListLengthIsChecked) {
  using V = zf::TensorF<3>;
  EXPECT_THROW((V{1.0f, 2.0f}), std::invalid_argument);
}

TEST(Tensor, Equality) {
  zf::TensorF<2> a{1, 2};
  zf::TensorF<2> b{1, 2};
  zf::TensorF<2> c{1, 3};
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(Tensor, ShapeToString) {
  zf::TensorF<2, 3> t;
  EXPECT_EQ(zf::shape_to_string(t), "[2, 3]");
}

TEST(Tensor, Int8Elements) {
  zf::TensorI8<3> t{-1, 2, 127};
  EXPECT_EQ(t(2), 127);
  EXPECT_EQ(sizeof(*t.data()), 1u);
}

TEST(Mul, ElementWise) {
  zf::TensorF<2, 2> a{1, 2, 3, 4};
  zf::TensorF<2, 2> b{5, 6, 7, 8};
  const zf::TensorF<2, 2> expected{5, 12, 21, 32};
  EXPECT_EQ(zf::mul(a, b), expected);
}

TEST(Mul, Scalar) {
  zf::TensorF<3> a{1, 2, 3};
  const zf::TensorF<3> expected{2, 4, 6};
  EXPECT_EQ(zf::mul(a, 2.0f), expected);
}

TEST(Mul, InPlace) {
  zf::TensorF<2> a{2, 3};
  const zf::TensorF<2> b{4, 5};
  zf::mul_into(a, b);
  EXPECT_FLOAT_EQ(a(0), 8.0f);
  EXPECT_FLOAT_EQ(a(1), 15.0f);
}

TEST(Mul, MatrixProduct) {
  // (2x3) * (3x2) -> (2x2)
  zf::TensorF<2, 3> a{1, 2, 3, 4, 5, 6};
  zf::TensorF<3, 2> b{7, 8, 9, 10, 11, 12};
  const zf::TensorF<2, 2> expected{58, 64, 139, 154};
  EXPECT_EQ(zf::matmul(a, b), expected);
}

TEST(Mul, MatrixTimesIdentity) {
  zf::TensorF<3, 3> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  zf::TensorF<3, 3> id;
  for (std::size_t i = 0; i < 3; ++i) {
    id(i, i) = 1.0f;
  }
  EXPECT_EQ(zf::matmul(a, id), a);
}

TEST(Mul, MatrixVectorProduct) {
  zf::TensorF<2, 3> a{1, 2, 3, 4, 5, 6};
  zf::TensorF<3> v{1, 0, 2};
  const zf::TensorF<2> expected{7, 16};
  EXPECT_EQ(zf::matmul(a, v), expected);
}

TEST(Mul, Int8MatmulWideDoesNotOverflow) {
  zf::TensorI8<2, 2> a{100, 100, 100, 100};
  zf::TensorI8<2, 2> b{100, 100, 100, 100};
  const auto wide = zf::matmul_wide(a, b);
  static_assert(
      std::is_same_v<std::decay_t<decltype(wide)>, zf::Tensor<std::int32_t, 2, 2>>);
  // 100*100 + 100*100 = 20000, well beyond int8 range.
  EXPECT_EQ(wide(0, 0), 20000);
}

TEST(Dot, Float) {
  zf::TensorF<3> a{1, 2, 3};
  zf::TensorF<3> b{4, 5, 6};
  EXPECT_FLOAT_EQ(zf::dot(a, b), 32.0f);
}

TEST(Dot, OverWholeMatrix) {
  zf::TensorF<2, 2> a{1, 2, 3, 4};
  zf::TensorF<2, 2> b{1, 1, 1, 1};
  EXPECT_FLOAT_EQ(zf::dot(a, b), 10.0f);
}

TEST(Dot, Int8AccumulatesInInt32) {
  zf::TensorI8<3> a{100, 100, 100};
  zf::TensorI8<3> b{100, 100, 100};
  const auto d = zf::dot(a, b);
  static_assert(std::is_same_v<std::decay_t<decltype(d)>, std::int32_t>);
  EXPECT_EQ(d, 30000);
}

TEST(Dot, SquaredNorm) {
  zf::TensorF<2> a{3, 4};
  EXPECT_FLOAT_EQ(zf::squared_norm(a), 25.0f);
}

TEST(Dot, Row) {
  zf::TensorF<3, 3> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  zf::TensorF<3> v{1, 1, 1};
  EXPECT_FLOAT_EQ(zf::dot_row(a, 1, v), 15.0f);
  EXPECT_THROW(zf::dot_row(a, 3, v), std::out_of_range);
}

}  // namespace
