#pragma once

#include <cstddef>

#include "tensor.h"

namespace zf {

// Element-wise product of two tensors of the same shape.
// Narrow element types (int8) multiply through the widened accumulator and are
// then truncated back to T, so the arithmetic itself does not depend on
// integer promotion rules.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...> mul(const Tensor<T, Dims...>& a,
                       const Tensor<T, Dims...>& b) {
  using Acc = accumulator_t<T>;
  Tensor<T, Dims...> out;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<T>(static_cast<Acc>(a[i]) * static_cast<Acc>(b[i]));
  }
  return out;
}

// Element-wise product by a scalar.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...> mul(const Tensor<T, Dims...>& a, T scalar) {
  using Acc = accumulator_t<T>;
  Tensor<T, Dims...> out;
  const Acc s = static_cast<Acc>(scalar);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<T>(static_cast<Acc>(a[i]) * s);
  }
  return out;
}

// In-place element-wise product; `a` is the destination.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...>& mul_into(Tensor<T, Dims...>& a,
                             const Tensor<T, Dims...>& b) {
  using Acc = accumulator_t<T>;
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<Acc>(a[i]) * static_cast<Acc>(b[i]));
  }
  return a;
}

// Matrix product (M x K) * (K x N) -> (M x N), accumulated in the widened
// accumulator and truncated back to T.
template <typename T, std::size_t M, std::size_t K, std::size_t N>
Tensor<T, M, N> matmul(const Tensor<T, M, K>& a, const Tensor<T, K, N>& b) {
  // i-k-j order: each a(i, k) is broadcast against a contiguous row of b, so
  // nothing walks down a column. Sums accumulate in a widened row first, so
  // int8 products still do not wrap.
  using Acc = accumulator_t<T>;
  Tensor<T, M, N> out;
  Acc row[N];
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) row[j] = Acc{};
    for (std::size_t k = 0; k < K; ++k) {
      const Acc aik = static_cast<Acc>(a(i, k));
      const T* bk = b.data() + k * N;
      for (std::size_t j = 0; j < N; ++j) row[j] += aik * static_cast<Acc>(bk[j]);
    }
    for (std::size_t j = 0; j < N; ++j) out(i, j) = static_cast<T>(row[j]);
  }
  return out;
}

// Same product, but the result keeps the widened accumulator type. For int8
// inputs this is the form that cannot overflow.
template <typename T, std::size_t M, std::size_t K, std::size_t N>
Tensor<accumulator_t<T>, M, N> matmul_wide(const Tensor<T, M, K>& a,
                                           const Tensor<T, K, N>& b) {
  using Acc = accumulator_t<T>;
  Tensor<Acc, M, N> out;
  for (std::size_t i = 0; i < M; ++i) {
    Acc* oi = out.data() + i * N;
    for (std::size_t k = 0; k < K; ++k) {
      const Acc aik = static_cast<Acc>(a(i, k));
      const T* bk = b.data() + k * N;
      for (std::size_t j = 0; j < N; ++j) oi[j] += aik * static_cast<Acc>(bk[j]);
    }
  }
  return out;
}

// Matrix-vector product (M x K) * (K) -> (M).
template <typename T, std::size_t M, std::size_t K>
Tensor<T, M> matmul(const Tensor<T, M, K>& a, const Tensor<T, K>& v) {
  using Acc = accumulator_t<T>;
  Tensor<T, M> out;
  for (std::size_t i = 0; i < M; ++i) {
    Acc sum{};
    for (std::size_t k = 0; k < K; ++k) {
      sum += static_cast<Acc>(a(i, k)) * static_cast<Acc>(v(k));
    }
    out(i) = static_cast<T>(sum);
  }
  return out;
}

}  // namespace zf
