#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "bf16.h"

// Non-aliasing hint for the arithmetic kernels. The tensors a kernel reads and
// the ones it writes are always distinct objects, but a compiler cannot prove
// that through two raw pointers, so without this it must
// reload after every store and cannot vectorize. Only used where the
// no-overlap precondition is stated in the kernel's comment.
#if defined(__clang__) || defined(__GNUC__)
#define ZF_RESTRICT __restrict
#else
#define ZF_RESTRICT
#endif

// The arithmetic kernels must inline into their caller's loop: left as calls,
// the accumulators are spilled to memory at every boundary and the whole point
// of holding sixteen of them is lost.
#if defined(__clang__) || defined(__GNUC__)
#define ZF_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ZF_ALWAYS_INLINE inline
#endif

namespace zf {

// Widened accumulator used by reductions so that int8 products do not wrap.
template <typename T>
struct accumulator {
  using type = T;
};
// bf16 is storage only: sums, and every intermediate in the layers, are float.
template <>
struct accumulator<bf16> {
  using type = float;
};
template <>
struct accumulator<std::int8_t> {
  using type = std::int32_t;
};
template <>
struct accumulator<std::uint8_t> {
  using type = std::uint32_t;
};
template <>
struct accumulator<std::int16_t> {
  using type = std::int32_t;
};
template <>
struct accumulator<std::uint16_t> {
  using type = std::uint32_t;
};

template <typename T>
using accumulator_t = typename accumulator<T>::type;

// Throws std::out_of_range; defined in tensor.cpp to keep the templates thin.
[[noreturn]] void throw_index_out_of_range(std::size_t axis, std::size_t index,
                                           std::size_t extent);
[[noreturn]] void throw_bad_element_count(std::size_t given,
                                          std::size_t expected);

// A run of real rows inside a padded, row-stacked tensor: rows [start,
// start + count). A batch of B padded sequences of length L is one tensor of
// B * L rows, and example b's real rows are the span {b * L, valid_b}. Layers
// that work row by row take a list of spans and touch only those rows.
struct RowSpan {
  std::size_t start = 0;
  std::size_t count = 0;
};
using RowSpans = std::span<const RowSpan>;

// Tensors up to this many bytes keep their elements inline, inside the object;
// bigger ones hold one heap block. Inline storage makes a temporary free -- no
// malloc/free per forward() -- and the shapes these models use are well under
// it. The cap keeps a stack frame holding a few temporaries reasonable.
inline constexpr std::size_t kInlineTensorBytes = 16 * 1024;

namespace detail {

// Contiguous, 64-byte aligned, zero-initialised storage for N elements.
template <typename T, std::size_t N, bool Inline = (N * sizeof(T) <= kInlineTensorBytes)>
class TensorStorage;

template <typename T, std::size_t N>
class TensorStorage<T, N, true> {
 public:
  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  alignas(64) T data_[N] = {};
};

// One heap block, allocated when the tensor is constructed and never again.
// Copies copy elements into the existing block; a move steals the block, and
// the moved-from tensor gets a fresh zeroed one, so it stays usable.
template <typename T, std::size_t N>
class TensorStorage<T, N, false> {
 public:
  TensorStorage() : data_(allocate()) {}
  TensorStorage(const TensorStorage& o) : data_(allocate()) {
    std::copy(o.data(), o.data() + N, data());
  }
  TensorStorage(TensorStorage&& o) noexcept : data_(std::move(o.data_)) {
    o.data_ = allocate();
  }
  TensorStorage& operator=(const TensorStorage& o) {
    if (this != &o) std::copy(o.data(), o.data() + N, data());
    return *this;
  }
  TensorStorage& operator=(TensorStorage&& o) noexcept {
    data_.swap(o.data_);
    return *this;
  }

  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }

 private:
  struct Free {
    void operator()(T* p) const { ::operator delete[](p, std::align_val_t{64}); }
  };
  static std::unique_ptr<T[], Free> allocate() {
    T* p = static_cast<T*>(::operator new[](N * sizeof(T), std::align_val_t{64}));
    std::fill(p, p + N, T{});
    return std::unique_ptr<T[], Free>(p);
  }
  std::unique_ptr<T[], Free> data_;
};

}  // namespace detail

// Fixed-shape, row-major tensor. The element type and every extent are
// template parameters, so the shape is known at compile time; the elements are
// one contiguous, 64-byte aligned block (see kInlineTensorBytes).
template <typename T, std::size_t... Dims>
class Tensor {
  static_assert(sizeof...(Dims) > 0, "tensor needs at least one dimension");
  static_assert(((Dims > 0) && ...), "tensor extents must be non-zero");

 public:
  using value_type = T;
  using accum_type = accumulator_t<T>;
  using iterator = T*;
  using const_iterator = const T*;

  static constexpr std::size_t rank = sizeof...(Dims);
  static constexpr std::size_t element_count = (std::size_t{1} * ... * Dims);
  static constexpr std::array<std::size_t, rank> shape = {Dims...};

  // Zero-filled.
  Tensor() = default;

  // Every element set to `value`.
  explicit Tensor(T value) { fill(value); }

  // Row-major list of all elements.
  Tensor(std::initializer_list<T> values) {
    if (values.size() != element_count) {
      throw_bad_element_count(values.size(), element_count);
    }
    std::copy(values.begin(), values.end(), begin());
  }

  static constexpr std::size_t size() { return element_count; }

  static constexpr std::size_t extent(std::size_t axis) {
    return shape[axis];
  }

  static constexpr std::array<std::size_t, rank> strides() {
    return strides_;
  }

  // --- multi-dimensional access (unchecked) ---

  template <typename... Idx>
  T& operator()(Idx... idx) {
    return data()[flat_index(idx...)];
  }

  template <typename... Idx>
  const T& operator()(Idx... idx) const {
    return data()[flat_index(idx...)];
  }

  // --- multi-dimensional access (bounds-checked) ---

  template <typename... Idx>
  T& at(Idx... idx) {
    return data()[checked_flat_index(idx...)];
  }

  template <typename... Idx>
  const T& at(Idx... idx) const {
    return data()[checked_flat_index(idx...)];
  }

  // --- flat access ---

  T& operator[](std::size_t i) { return data()[i]; }
  const T& operator[](std::size_t i) const { return data()[i]; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

  iterator begin() { return data(); }
  iterator end() { return data() + element_count; }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + element_count; }

  void fill(T value) { std::fill(begin(), end(), value); }

  friend bool operator==(const Tensor& a, const Tensor& b) {
    return std::equal(a.begin(), a.end(), b.begin());
  }
  friend bool operator!=(const Tensor& a, const Tensor& b) {
    return !(a == b);
  }

  // Row-major offset of `idx...`, no bounds checking.
  template <typename... Idx>
  static constexpr std::size_t flat_index(Idx... idx) {
    static_assert(sizeof...(Idx) == rank,
                  "number of indices must match tensor rank");
    const std::array<std::size_t, rank> ii = {
        static_cast<std::size_t>(idx)...};
    std::size_t offset = 0;
    for (std::size_t a = 0; a < rank; ++a) {
      offset += ii[a] * strides_[a];
    }
    return offset;
  }

 private:
  template <typename... Idx>
  static std::size_t checked_flat_index(Idx... idx) {
    static_assert(sizeof...(Idx) == rank,
                  "number of indices must match tensor rank");
    const std::array<std::size_t, rank> ii = {
        static_cast<std::size_t>(idx)...};
    std::size_t offset = 0;
    for (std::size_t a = 0; a < rank; ++a) {
      if (ii[a] >= shape[a]) {
        throw_index_out_of_range(a, ii[a], shape[a]);
      }
      offset += ii[a] * strides_[a];
    }
    return offset;
  }

  static constexpr std::array<std::size_t, rank> make_strides() {
    std::array<std::size_t, rank> s{};
    std::size_t acc = 1;
    for (std::size_t a = rank; a-- > 0;) {
      s[a] = acc;
      acc *= shape[a];
    }
    return s;
  }

  static constexpr std::array<std::size_t, rank> strides_ = make_strides();

  detail::TensorStorage<T, element_count> data_;
};

// "[2, 3]" — handy in assertions and error messages.
std::string shape_to_string(const std::size_t* dims, std::size_t rank);

template <typename T, std::size_t... Dims>
std::string shape_to_string(const Tensor<T, Dims...>& t) {
  return shape_to_string(t.shape.data(), t.rank);
}

// True when no element is NaN or infinite.
template <typename T, std::size_t... Dims>
bool all_finite(const Tensor<T, Dims...>& t) {
  if constexpr (is_real_v<T>) {
    for (const T v : t) {
      if (!std::isfinite(static_cast<accumulator_t<T>>(v))) return false;
    }
  }
  return true;
}

// Element-wise conversion, e.g. a float tensor into bf16 (rounding to nearest
// even) or back (exact).
template <typename To, typename From, std::size_t... Dims>
Tensor<To, Dims...> tensor_cast(const Tensor<From, Dims...>& t) {
  Tensor<To, Dims...> out;
  for (std::size_t i = 0; i < t.size(); ++i) {
    out[i] = static_cast<To>(static_cast<accumulator_t<From>>(t[i]));
  }
  return out;
}

// `t` as a tensor of To: the same object when it already is one, a converted
// copy otherwise. For code that keeps a float scratch tensor and must hand a
// T tensor on, at no cost when T is float.
template <typename To, typename From, std::size_t... Dims>
decltype(auto) as_tensor_of(const Tensor<From, Dims...>& t) {
  if constexpr (std::is_same_v<To, From>) {
    return (t);
  } else {
    return tensor_cast<To>(t);
  }
}

// --- small tensor helpers ----------------------------------------------------

template <typename T, std::size_t N>
Tensor<T, N> to_tensor(const std::array<float, N>& a) {
  Tensor<T, N> t;
  for (std::size_t i = 0; i < N; ++i) t(i) = static_cast<T>(a[i]);
  return t;
}

template <typename T, std::size_t R, std::size_t D>
Tensor<T, D> row_of(const Tensor<T, R, D>& m, std::size_t r) {
  Tensor<T, D> v;
  for (std::size_t d = 0; d < D; ++d) v(d) = m(r, d);
  return v;
}

template <typename T, std::size_t R, std::size_t D>
void add_row(Tensor<T, R, D>* m, std::size_t r, const Tensor<T, D>& v) {
  for (std::size_t d = 0; d < D; ++d) (*m)(r, d) += v(d);
}

template <typename T, std::size_t R, std::size_t D>
void put_row(Tensor<T, R, D>* m, std::size_t r, const Tensor<T, D>& v) {
  for (std::size_t d = 0; d < D; ++d) (*m)(r, d) = v(d);
}

[[noreturn]] void throw_not_finite(const char* what);

// Debug check after each major step: with ZTENSOR_CHECK_FINITE defined (the
// CMake option of the same name), a NaN or infinity in `t` throws
// std::runtime_error naming `what`, at the step that produced it instead of
// several layers later. Compiles to nothing otherwise.
#if defined(ZTENSOR_CHECK_FINITE)
#define ZF_CHECK_FINITE(t, what)                  \
  do {                                            \
    if (!::zf::all_finite(t)) ::zf::throw_not_finite(what); \
  } while (0)
#else
#define ZF_CHECK_FINITE(t, what) \
  do {                           \
  } while (0)
#endif

template <std::size_t... Dims>
using TensorF = Tensor<float, Dims...>;

template <std::size_t... Dims>
using TensorB = Tensor<bf16, Dims...>;

template <std::size_t... Dims>
using TensorI8 = Tensor<std::int8_t, Dims...>;

}  // namespace zf
