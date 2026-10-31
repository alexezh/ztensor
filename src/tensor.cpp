#include "tensor.h"

namespace zf {

void throw_index_out_of_range(std::size_t axis, std::size_t index,
                              std::size_t extent) {
  throw std::out_of_range("tensor index " + std::to_string(index) +
                          " out of range on axis " + std::to_string(axis) +
                          " (extent " + std::to_string(extent) + ")");
}

void throw_bad_element_count(std::size_t given, std::size_t expected) {
  throw std::invalid_argument(
      "tensor initializer has " + std::to_string(given) +
      " elements, expected " + std::to_string(expected));
}

std::string shape_to_string(const std::size_t* dims, std::size_t rank) {
  std::string out = "[";
  for (std::size_t i = 0; i < rank; ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(dims[i]);
  }
  out += "]";
  return out;
}

// Instantiate a few common shapes so this translation unit type-checks them.
template class Tensor<float, 3>;
template class Tensor<float, 2, 2>;
template class Tensor<float, 3, 3>;
template class Tensor<float, 4, 4>;
template class Tensor<std::int8_t, 3>;
template class Tensor<std::int8_t, 2, 2>;
template class Tensor<std::int8_t, 3, 3>;

}  // namespace zf
