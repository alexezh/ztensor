#include "dot.h"

namespace zf {

// Explicit instantiations for the common element types and shapes.
template float dot(const TensorF<3>&, const TensorF<3>&);
template float dot(const TensorF<4>&, const TensorF<4>&);
template float dot(const TensorF<3, 3>&, const TensorF<3, 3>&);
template float squared_norm(const TensorF<3>&);
template float dot_row(const TensorF<3, 3>&, std::size_t, const TensorF<3>&);

template std::int32_t dot(const TensorI8<3>&, const TensorI8<3>&);
template std::int32_t dot(const TensorI8<2, 2>&, const TensorI8<2, 2>&);
template std::int32_t squared_norm(const TensorI8<3>&);

}  // namespace zf
