#pragma once

#include <cstddef>

// Optional BLAS backend for Linear<float>'s multi-row paths. Built when
// ZTENSOR_ACCELERATE is on (the default on Apple), which defines
// ZF_USE_ACCELERATE and links Apple's Accelerate framework.
//
// These are plain functions defined in blas.cpp, so the Accelerate header,
// with its macros and its weight, stays out of every user translation unit.
// All matrices are row-major and densely packed; the calls are tiny next to
// the GEMM behind them.

namespace zf::blas {

#if defined(ZF_USE_ACCELERATE)
inline constexpr bool kEnabled = true;
#else
inline constexpr bool kEnabled = false;
#endif

// The fewest rows sent to the GEMM; fewer go to the inline kernels. Measured
// on Apple Silicon at widths 32-64:
//  - below 16 rows SGEMM takes a slow path and often loses outright;
//  - in a tight loop it wins 2-3x from 16 rows up, but between other work
//    (an attention layer, say) the matrix unit cools down and a 48-row call
//    gains little, while attention around it gets slower;
//  - a row-stacked batch (64 examples x 12 rows) gains robustly.
// So the GEMM is for batches, and a single sequence stays on the kernels.
inline constexpr std::size_t kMinRows = 64;

// out[rows, n] = x[rows, k] * w[n, k]^T + b[n]. Overwrites out.
void linear_forward(const float* x, const float* w, const float* b, float* out,
                    std::size_t rows, std::size_t k, std::size_t n);

// For y = x w^T + b with incoming gradient g[rows, n]:
//   dx[rows, k] += g w,  dw[n, k] += g^T x,  db[n] += column sums of g.
void linear_backward(const float* x, const float* g, const float* w, float* dx,
                     float* dw, float* db, std::size_t rows, std::size_t k,
                     std::size_t n);

}  // namespace zf::blas
