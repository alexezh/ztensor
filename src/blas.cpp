#include "blas.h"

#if defined(ZF_USE_ACCELERATE)

#include <algorithm>
#include <atomic>

#include <sys/sysctl.h>

#include "thread_pool.h"

// The new interface: without it, macOS 13.3+ marks every cblas_* deprecated.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

namespace zf::blas {

namespace {
using Int = __LAPACK_int;
Int to_int(std::size_t v) { return static_cast<Int>(v); }

std::atomic<ThreadPool*> g_pool{nullptr};
std::atomic<bool> g_pool_set{false};

int sysctl_int(const char* name) {
  int v = 0;
  std::size_t size = sizeof(v);
  return sysctlbyname(name, &v, &size, nullptr, 0) == 0 ? v : 0;
}

// Parts for a call of this size: one per matrix unit, when there is enough
// work for each, and never more than the pool has threads.
std::size_t parts_for(std::size_t rows, std::size_t k, std::size_t n, std::size_t limit) {
  ThreadPool* p = pool();
  if (p == nullptr || ThreadPool::on_worker() || rows * k * n < kSplitWork) return 1;
  std::size_t parts = std::min({matrix_units(), p->participants(), limit});
  return parts < 1 ? 1 : parts;
}

void forward_part(const float* x, const float* w, const float* b, float* out, std::size_t rows,
                  std::size_t k, std::size_t n) {
  // Seed every row with the bias and let the GEMM add into it (beta = 1).
  for (std::size_t r = 0; r < rows; ++r) std::copy(b, b + n, out + r * n);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, to_int(rows), to_int(n),
              to_int(k), 1.0f, x, to_int(k), w, to_int(k), 1.0f, out, to_int(n));
}

// dx rows [r0, r1) += g w.
void backward_dx(const float* g, const float* w, float* dx, std::size_t r0, std::size_t r1,
                 std::size_t k, std::size_t n) {
  if (r1 == r0) return;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, to_int(r1 - r0), to_int(k),
              to_int(n), 1.0f, g + r0 * n, to_int(n), w, to_int(k), 1.0f, dx + r0 * k, to_int(k));
}

// dw rows [o0, o1) += g[:, o0:o1]^T x, db[o0:o1] += column sums of g.
void backward_dw(const float* x, const float* g, float* dw, float* db, std::size_t rows,
                 std::size_t o0, std::size_t o1, std::size_t k, std::size_t n) {
  if (o1 == o0) return;
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, to_int(o1 - o0), to_int(k),
              to_int(rows), 1.0f, g + o0, to_int(n), x, to_int(k), 1.0f, dw + o0 * k, to_int(k));
  // Row by row in order, as on one thread.
  for (std::size_t r = 0; r < rows; ++r) {
    const float* gr = g + r * n;
    for (std::size_t o = o0; o < o1; ++o) db[o] += gr[o];
  }
}
}  // namespace

std::size_t matrix_units() {
  static const std::size_t units = [] {
    const int cores = sysctl_int("hw.perflevel0.physicalcpu");
    const int per_l2 = sysctl_int("hw.perflevel0.cpusperl2");
    const int clusters = cores > 0 && per_l2 > 0 ? (cores + per_l2 - 1) / per_l2 : 1;
    return static_cast<std::size_t>(clusters > 0 ? clusters : 1);
  }();
  return units;
}

void set_pool(ThreadPool* p) {
  g_pool.store(p, std::memory_order_release);
  g_pool_set.store(true, std::memory_order_release);
}

ThreadPool* pool() {
  if (!g_pool_set.load(std::memory_order_acquire)) {
    // Its own pool, one thread per matrix unit: a part queued to the
    // global pool's dozen threads lands as often on an efficiency core, or
    // next to the caller on its cluster, and gains nothing.
    // ZF_THREADS caps it too (ZF_THREADS=1: no helper at all).
    static ThreadPool own(std::min(matrix_units(), ThreadPool::default_participants()));
    return &own;
  }
  return g_pool.load(std::memory_order_acquire);
}

void linear_forward(const float* x, const float* w, const float* b, float* out,
                    std::size_t rows, std::size_t k, std::size_t n) {
  const std::size_t parts = parts_for(rows, k, n, rows / kMinRows);
  if (parts <= 1) {
    forward_part(x, w, b, out, rows, k, n);
    return;
  }
  parallel_for(pool(), rows, (rows + parts - 1) / parts,
               [&](std::size_t, std::size_t r0, std::size_t r1) {
                 forward_part(x + r0 * k, w, b, out + r0 * n, r1 - r0, k, n);
               });
}

void linear_backward(const float* x, const float* g, const float* w, float* dx,
                     float* dw, float* db, std::size_t rows, std::size_t k,
                     std::size_t n) {
  const std::size_t parts = parts_for(rows, k, n, std::min(rows / kMinRows, n));
  if (parts <= 1) {
    backward_dx(g, w, dx, 0, rows, k, n);
    backward_dw(x, g, dw, db, rows, 0, n, k, n);
    return;
  }
  // Part c: its rows of dx and its output columns of dw, db.
  parallel_for(pool(), parts, 1, [&](std::size_t, std::size_t c0, std::size_t c1) {
    for (std::size_t c = c0; c < c1; ++c) {
      backward_dx(g, w, dx, chunk_begin(rows, parts, c), chunk_begin(rows, parts, c + 1), k, n);
      backward_dw(x, g, dw, db, rows, chunk_begin(n, parts, c), chunk_begin(n, parts, c + 1), k, n);
    }
  });
}

}  // namespace zf::blas

#endif  // ZF_USE_ACCELERATE
