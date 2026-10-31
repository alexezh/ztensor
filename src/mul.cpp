#include "mul.h"

namespace zf {

// Explicit instantiations: these give the templates a translation unit of
// their own and keep the common shapes type-checked at library build time.
template TensorF<3, 3> mul(const TensorF<3, 3>&, const TensorF<3, 3>&);
template TensorF<3, 3> mul(const TensorF<3, 3>&, float);
template TensorF<3, 3>& mul_into(TensorF<3, 3>&, const TensorF<3, 3>&);
template TensorF<3, 3> matmul(const TensorF<3, 3>&, const TensorF<3, 3>&);
template TensorF<3> matmul(const TensorF<3, 3>&, const TensorF<3>&);

template TensorF<4, 4> matmul(const TensorF<4, 4>&, const TensorF<4, 4>&);

template TensorI8<2, 2> mul(const TensorI8<2, 2>&, const TensorI8<2, 2>&);
template TensorI8<2, 2> matmul(const TensorI8<2, 2>&, const TensorI8<2, 2>&);
template Tensor<std::int32_t, 2, 2> matmul_wide(const TensorI8<2, 2>&,
                                                const TensorI8<2, 2>&);

}  // namespace zf
