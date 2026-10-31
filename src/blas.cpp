#include "blas.h"

#if defined(ZF_USE_ACCELERATE)

#include <algorithm>

// The new interface: without it, macOS 13.3+ marks every cblas_* deprecated.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

namespace zf::blas {

namespace {
using Int = __LAPACK_int;
Int to_int(std::size_t v) { return static_cast<Int>(v); }
}  // namespace

void linear_forward(const float* x, const float* w, const float* b, float* out,
                    std::size_t rows, std::size_t k, std::size_t n) {
  // Seed every row with the bias and let the GEMM add into it (beta = 1).
  for (std::size_t r = 0; r < rows; ++r) std::copy(b, b + n, out + r * n);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, to_int(rows), to_int(n),
              to_int(k), 1.0f, x, to_int(k), w, to_int(k), 1.0f, out, to_int(n));
}

void linear_backward(const float* x, const float* g, const float* w, float* dx,
                     float* dw, float* db, std::size_t rows, std::size_t k,
                     std::size_t n) {
  // dx += g w
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, to_int(rows), to_int(k),
              to_int(n), 1.0f, g, to_int(n), w, to_int(k), 1.0f, dx, to_int(k));
  // dw += g^T x
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, to_int(n), to_int(k),
              to_int(rows), 1.0f, g, to_int(n), x, to_int(k), 1.0f, dw, to_int(k));
  // db += column sums of g, row by row in order.
  for (std::size_t r = 0; r < rows; ++r) {
    const float* gr = g + r * n;
    for (std::size_t o = 0; o < n; ++o) db[o] += gr[o];
  }
}

}  // namespace zf::blas

#endif  // ZF_USE_ACCELERATE
