#pragma once

#include <cstddef>

#include "tensor.h"

namespace zf {

// Sum of element-wise products over the whole tensor. Both operands must have
// the same shape; the result uses the widened accumulator type, so an int8 dot
// product comes back as int32.
template <typename T, std::size_t... Dims>
accumulator_t<T> dot(const Tensor<T, Dims...>& a,
                     const Tensor<T, Dims...>& b) {
  using Acc = accumulator_t<T>;
  Acc sum{};
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<Acc>(a[i]) * static_cast<Acc>(b[i]);
  }
  return sum;
}

// Sum of squares, i.e. dot(a, a).
template <typename T, std::size_t... Dims>
accumulator_t<T> squared_norm(const Tensor<T, Dims...>& a) {
  return dot(a, a);
}

// Dot product of one row of `a` with `v`: sum_k a(row, k) * v(k).
template <typename T, std::size_t M, std::size_t K>
accumulator_t<T> dot_row(const Tensor<T, M, K>& a, std::size_t row,
                         const Tensor<T, K>& v) {
  using Acc = accumulator_t<T>;
  Acc sum{};
  for (std::size_t k = 0; k < K; ++k) {
    sum += static_cast<Acc>(a.at(row, k)) * static_cast<Acc>(v(k));
  }
  return sum;
}

}  // namespace zf
