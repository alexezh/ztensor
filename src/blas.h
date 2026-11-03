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

namespace zf {
class ThreadPool;
}

namespace zf::blas {

#if defined(ZF_USE_ACCELERATE)
inline constexpr bool kEnabled = true;
#else
inline constexpr bool kEnabled = false;
#endif

// When a Linear call goes to the GEMM rather than the inline kernels: at
// least kMinRows rows, and at least kMinWork multiply-adds (rows x k x n).
//
// The bar is the work, not the rows. Measured on Apple Silicon:
//  - In a tight loop SGEMM ties or wins from 8 rows at every width tried
//    (32x32 up to 354x128), and by 3-10x at 16-32 rows of a 354-wide input.
//    Below 4 rows its backward takes a slow path: 5x the kernels at 2 rows.
//  - Between other work it is different. At width 32 (this library's
//    MultiHeadSelfAttention, 48 rows; a 12-row decoder) a Linear alone gains
//    2.3x, but the attention around it slows, and the blocks come out 5-18%
//    slower: there is too little work per call to pay for the switch.
//  - With wide inputs it pays from a few rows. A 4-level cross transformer
//    on 354-wide tokens, one example per call (8-14 rows), trained 1.5x
//    faster under this rule than with a flat 64-row bar.
// 128K multiply-adds keeps a 32x32 layer on the kernels below 128 rows (and
// a 32x64 one below 64), and sends a 354x64 one from 6 rows. Half that bar
// gains the wide model another 8% but costs this library's TransformerBlock
// forward 10%.
inline constexpr std::size_t kMinRows = 4;
inline constexpr std::size_t kMinWork = std::size_t{128} * 1024;

constexpr bool use_gemm(std::size_t rows, std::size_t k, std::size_t n) {
  return rows >= kMinRows && rows * k * n >= kMinWork;
}

// The fewest rows of a k -> n layer that go to the GEMM.
constexpr std::size_t gemm_rows(std::size_t k, std::size_t n) {
  const std::size_t by_work = (kMinWork + k * n - 1) / (k * n);
  return by_work > kMinRows ? by_work : kMinRows;
}

// Large calls split over the matrix units, one part each. Apple Silicon has
// one per performance cluster (an M2 Max two, an M2 one, an Ultra four); a
// second thread on the same cluster shares its unit and only adds
// contention. Measured on an M2 Max, 224 rows of 354 -> 64: one call 21 us,
// two parts on two threads 13 us, four or eight parts 20 us or worse.
// Forward splits the rows; backward splits dx by rows and dw, db by output
// columns, so no part adds into another's sums and each element is still one
// GEMM. The parts run on `pool()`: unless set, a pool of its own with one
// thread per matrix unit (with ThreadPool::global()'s dozen threads a part
// lands as often on an efficiency core, or beside the caller, and the
// split gains nothing).
inline constexpr std::size_t kSplitWork = std::size_t{2} * 1024 * 1024;

std::size_t matrix_units();
// nullptr: never split.
void set_pool(ThreadPool* pool);
ThreadPool* pool();

// out[rows, n] = x[rows, k] * w[n, k]^T + b[n]. Overwrites out.
void linear_forward(const float* x, const float* w, const float* b, float* out,
                    std::size_t rows, std::size_t k, std::size_t n);

// For y = x w^T + b with incoming gradient g[rows, n]:
//   dx[rows, k] += g w,  dw[n, k] += g^T x,  db[n] += column sums of g.
void linear_backward(const float* x, const float* g, const float* w, float* dx,
                     float* dw, float* db, std::size_t rows, std::size_t k,
                     std::size_t n);

}  // namespace zf::blas
