#pragma once

#include <cstddef>

#include "tensor.h"

namespace zf {

// Element-wise sum of two tensors of the same shape.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...> add(const Tensor<T, Dims...>& a,
                       const Tensor<T, Dims...>& b) {
  using Acc = accumulator_t<T>;
  Tensor<T, Dims...> out;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<T>(static_cast<Acc>(a[i]) + static_cast<Acc>(b[i]));
  }
  return out;
}

// Element-wise difference, a - b.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...> sub(const Tensor<T, Dims...>& a,
                       const Tensor<T, Dims...>& b) {
  using Acc = accumulator_t<T>;
  Tensor<T, Dims...> out;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<T>(static_cast<Acc>(a[i]) - static_cast<Acc>(b[i]));
  }
  return out;
}

// a += b, in place.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...>& add_into(Tensor<T, Dims...>& a,
                             const Tensor<T, Dims...>& b) {
  using Acc = accumulator_t<T>;
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<Acc>(a[i]) + static_cast<Acc>(b[i]));
  }
  return a;
}

// a += alpha * b, in place. The workhorse for gradient accumulation and SGD.
template <typename T, std::size_t... Dims>
Tensor<T, Dims...>& add_scaled_into(Tensor<T, Dims...>& a,
                                    const Tensor<T, Dims...>& b, T alpha) {
  using Acc = accumulator_t<T>;
  const Acc s = static_cast<Acc>(alpha);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<Acc>(a[i]) + s * static_cast<Acc>(b[i]));
  }
  return a;
}

}  // namespace zf
