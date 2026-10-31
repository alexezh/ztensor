#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

// Non-aliasing hint for the arithmetic kernels. The tensors a kernel reads and
// the ones it writes are always distinct objects, but a compiler cannot prove
// that through two pointers into two std::vectors, so without this it must
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

// Fixed-shape, row-major tensor for small sizes. The element type and every
// extent are template parameters, so the shape is known at compile time; the
// elements live in a single std::vector.
template <typename T, std::size_t... Dims>
class Tensor {
  static_assert(sizeof...(Dims) > 0, "tensor needs at least one dimension");
  static_assert(((Dims > 0) && ...), "tensor extents must be non-zero");

 public:
  using value_type = T;
  using accum_type = accumulator_t<T>;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  static constexpr std::size_t rank = sizeof...(Dims);
  static constexpr std::size_t element_count = (std::size_t{1} * ... * Dims);
  static constexpr std::array<std::size_t, rank> shape = {Dims...};

  // Zero-filled.
  Tensor() : data_(element_count, T{}) {}

  // Every element set to `value`.
  explicit Tensor(T value) : data_(element_count, value) {}

  // Row-major list of all elements.
  Tensor(std::initializer_list<T> values) : data_(element_count, T{}) {
    if (values.size() != element_count) {
      throw_bad_element_count(values.size(), element_count);
    }
    std::size_t i = 0;
    for (T v : values) {
      data_[i++] = v;
    }
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
    return data_[flat_index(idx...)];
  }

  template <typename... Idx>
  const T& operator()(Idx... idx) const {
    return data_[flat_index(idx...)];
  }

  // --- multi-dimensional access (bounds-checked) ---

  template <typename... Idx>
  T& at(Idx... idx) {
    return data_[checked_flat_index(idx...)];
  }

  template <typename... Idx>
  const T& at(Idx... idx) const {
    return data_[checked_flat_index(idx...)];
  }

  // --- flat access ---

  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

  iterator begin() { return data_.begin(); }
  iterator end() { return data_.end(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }

  void fill(T value) {
    for (T& v : data_) {
      v = value;
    }
  }

  friend bool operator==(const Tensor& a, const Tensor& b) {
    return a.data_ == b.data_;
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

  std::vector<T> data_;
};

// "[2, 3]" — handy in assertions and error messages.
std::string shape_to_string(const std::size_t* dims, std::size_t rank);

template <typename T, std::size_t... Dims>
std::string shape_to_string(const Tensor<T, Dims...>& t) {
  return shape_to_string(t.shape.data(), t.rank);
}

template <std::size_t... Dims>
using TensorF = Tensor<float, Dims...>;

template <std::size_t... Dims>
using TensorI8 = Tensor<std::int8_t, Dims...>;

}  // namespace zf
