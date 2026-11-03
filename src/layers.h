#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

#include "blas.h"
#include "tensor.h"
#include "thread_pool.h"

// Primitive layers shared by the models.
//
// Convention: primitives are stateless with respect to activations. `forward`
// is const, and `backward(x, dOut)` is handed the same input `x` that produced
// the output; it accumulates parameter gradients and returns dL/dx. The
// composite models (Mlp, TransformerBlock, QueryModel, ...) are the ones that
// cache activations, because only they know every intermediate shape.

namespace zf {

enum class Activation { kIdentity, kRelu, kGelu };

inline float softmax_exp(float x);  // defined below, with the softmax

// tanh for the GELU. std::tanh is a library call per element, so the loop
// around it cannot vectorize; in a training step it was about 13% of the
// time. tanh is odd, and for a = |u|
//   tanh(a) = (1 - e) / (1 + e),  e = exp(-2a) in (0, 1]
// which needs exp only of non-positive arguments -- exactly the range
// softmax_exp covers, to about 3 ulp. Branch-free, so it vectorizes.
// Other element types keep std::tanh.
template <typename T>
inline T fast_tanh(T u) {
  if constexpr (std::is_same_v<T, float>) {
    const float e = softmax_exp(-2.0f * std::fabs(u));
    return std::copysign((1.0f - e) / (1.0f + e), u);
  } else {
    return std::tanh(u);
  }
}

template <typename T>
T activation_apply(Activation act, T x) {
  switch (act) {
    case Activation::kRelu:
      return x > T{0} ? x : T{0};
    case Activation::kGelu: {
      // tanh approximation of GELU. u is written exactly as in
      // activation_grad and activation_apply_with_grad, so all three round
      // the same way.
      const T c = static_cast<T>(0.7978845608028654);  // sqrt(2/pi)
      const T x2 = x * x;
      const T u = c * (x + static_cast<T>(0.044715) * x * x2);
      return static_cast<T>(0.5) * x * (T{1} + fast_tanh(u));
    }
    case Activation::kIdentity:
      break;
  }
  return x;
}

// Derivative of the activation with respect to its pre-activation input.
template <typename T>
T activation_grad(Activation act, T x) {
  switch (act) {
    case Activation::kRelu:
      return x > T{0} ? T{1} : T{0};
    case Activation::kGelu: {
      const T c = static_cast<T>(0.7978845608028654);
      const T x2 = x * x;
      const T u = c * (x + static_cast<T>(0.044715) * x * x2);
      const T t = fast_tanh(u);
      const T du = c * (T{1} + static_cast<T>(3 * 0.044715) * x2);
      return static_cast<T>(0.5) * (T{1} + t) +
             static_cast<T>(0.5) * x * (T{1} - t * t) * du;
    }
    case Activation::kIdentity:
      break;
  }
  return T{1};
}

// Activation and its derivative together, sharing the one tanh. A layer that
// will need the derivative in backward stores it here instead of recomputing.
// `Act` is a template parameter, so the choice is made at compile time and the
// inner loop carries no switch.
template <Activation Act, typename T>
ZF_ALWAYS_INLINE T activate_with_grad(T x, T* grad) {
  if constexpr (Act == Activation::kRelu) {
    *grad = x > T{0} ? T{1} : T{0};
    return x > T{0} ? x : T{0};
  } else if constexpr (Act == Activation::kGelu) {
    const T c = static_cast<T>(0.7978845608028654);
    const T x2 = x * x;
    const T u = c * (x + static_cast<T>(0.044715) * x * x2);
    const T t = fast_tanh(u);
    const T du = c * (T{1} + static_cast<T>(3 * 0.044715) * x2);
    *grad = static_cast<T>(0.5) * (T{1} + t) +
            static_cast<T>(0.5) * x * (T{1} - t * t) * du;
    return static_cast<T>(0.5) * x * (T{1} + t);
  } else {
    *grad = T{1};
    return x;
  }
}

// The same, with the activation chosen at run time.
template <typename T>
T activation_apply_with_grad(Activation act, T x, T* grad) {
  switch (act) {
    case Activation::kRelu:
      return activate_with_grad<Activation::kRelu>(x, grad);
    case Activation::kGelu:
      return activate_with_grad<Activation::kGelu>(x, grad);
    case Activation::kIdentity:
      break;
  }
  return activate_with_grad<Activation::kIdentity>(x, grad);
}

// bf16: the same functions evaluated in float, rounded once on the way out.
inline bf16 activation_apply(Activation act, bf16 x) {
  return bf16(activation_apply(act, static_cast<float>(x)));
}
inline bf16 activation_grad(Activation act, bf16 x) {
  return bf16(activation_grad(act, static_cast<float>(x)));
}
template <Activation Act>
ZF_ALWAYS_INLINE bf16 activate_with_grad(bf16 x, bf16* grad) {
  float g;
  const float y = activate_with_grad<Act>(static_cast<float>(x), &g);
  *grad = bf16(g);
  return bf16(y);
}

// y = W x + b, with W stored as [OutDim, InDim].
// A 4-wide SIMD value for the kernels: NEON / SSE registers through the
// compiler's vector extension, plain arrays elsewhere.
#if defined(__clang__) || defined(__GNUC__)
template <typename T>
using Vec4 = T __attribute__((vector_size(4 * sizeof(T))));
#else
template <typename T>
struct Vec4 {
  T v[4] = {};
  T& operator[](std::size_t i) { return v[i]; }
  T operator[](std::size_t i) const { return v[i]; }
  Vec4& operator+=(const Vec4& o) {
    for (std::size_t i = 0; i < 4; ++i) v[i] += o.v[i];
    return *this;
  }
  friend Vec4 operator+(Vec4 a, const Vec4& b) { return a += b; }
  friend Vec4 operator*(Vec4 a, const Vec4& b) {
    for (std::size_t i = 0; i < 4; ++i) a.v[i] *= b.v[i];
    return a;
  }
  friend Vec4 operator*(T s, Vec4 a) {
    for (std::size_t i = 0; i < 4; ++i) a.v[i] *= s;
    return a;
  }
};
#endif

// Four elements into a vector of the math type: a plain load for float, a
// widening load for bf16 (each lane's 16 bits shifted to the top of a float,
// which is exact and one instruction on NEON and SSE).
template <typename T>
ZF_ALWAYS_INLINE Vec4<accumulator_t<T>> load4(const T* p) {
  Vec4<accumulator_t<T>> v;
  if constexpr (std::is_same_v<T, bf16>) {
#if defined(__clang__) || defined(__GNUC__)
    using U16x4 = std::uint16_t __attribute__((vector_size(8)));
    using U32x4 = std::uint32_t __attribute__((vector_size(16)));
    U16x4 h;
    std::memcpy(&h, p, sizeof(h));
    const U32x4 w = __builtin_convertvector(h, U32x4) << 16;
    std::memcpy(&v, &w, sizeof(v));
#else
    for (std::size_t i = 0; i < 4; ++i) v[i] = static_cast<float>(p[i]);
#endif
  } else {
    std::memcpy(&v, p, sizeof(v));
  }
  return v;
}

template <typename T>
ZF_ALWAYS_INLINE void store4(T* p, const Vec4<T>& v) {
  std::memcpy(p, &v, sizeof(v));
}

template <typename T>
ZF_ALWAYS_INLINE T hsum4(const Vec4<T>& v) {
  return (v[0] + v[1]) + (v[2] + v[3]);
}

// n elements of T into the math type, and back with rounding. Plain loops:
// both vectorize (widening is a shift, rounding an add and a shift).
template <typename T>
inline void widen_into(const T* ZF_RESTRICT src, accumulator_t<T>* ZF_RESTRICT dst,
                       std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<accumulator_t<T>>(src[i]);
}

template <typename T>
inline void narrow_into(const accumulator_t<T>* ZF_RESTRICT src, T* ZF_RESTRICT dst,
                        std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i]);
}

namespace detail {

// A per-thread float buffer of at least n elements; `slot` keeps the buffers
// one call needs at the same time apart. Grows, never shrinks.
template <typename A>
inline A* widen_scratch(std::size_t slot, std::size_t n) {
  thread_local std::vector<A> buffers[4];
  std::vector<A>& b = buffers[slot];
  if (b.size() < n) b.resize(n);
  return b.data();
}

}  // namespace detail

template <typename T, std::size_t InDim, std::size_t OutDim>
class Linear {
  static_assert(is_real_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  // The math type: float for both float and bf16. Parameters are stored as
  // T; their gradients, which accumulate over every row of a batch, as Acc.
  using Acc = accumulator_t<T>;
  using WeightTensor = Tensor<T, OutDim, InDim>;
  using BiasTensor = Tensor<T, OutDim>;
  using WeightGradTensor = Tensor<Acc, OutDim, InDim>;
  using BiasGradTensor = Tensor<Acc, OutDim>;

  static constexpr std::size_t in_dim = InDim;
  static constexpr std::size_t out_dim = OutDim;
  static constexpr std::size_t parameter_count = OutDim * InDim + OutDim;

  Linear() = default;

  // He-style normal init; bias zeroed.
  template <typename Rng>
  void randomize(Rng& rng) {
    randomize(rng, static_cast<T>(std::sqrt(2.0 / static_cast<double>(InDim))));
  }

  template <typename Rng>
  void randomize(Rng& rng, T stddev) {
    std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));
    for (T& v : w_) {
      v = static_cast<T>(dist(rng));
    }
    b_.fill(T{});
  }

  // --- the kernels ---
  //
  // These do all the arithmetic in the library: every projection, every
  // feed-forward layer and every head is a call into one of them. They are
  // written for the machine rather than for the page, so the shape is
  // explained here.
  //
  // Three things cost time in the obvious triple loop.
  //
  // One: `sum += w[i] * x[i]` is a chain where every multiply-add waits for
  // the one before it, so the loop runs at the latency of an FMA instead of
  // its throughput. Reassociating floating-point addition changes the result,
  // so a compiler will not split the chain unless told it may. A 4-wide
  // vector accumulator is four independent chains.
  //
  // Two: computing one output at a time re-reads the input row for each row
  // of W. A block of 4 outputs reads the input once.
  //
  // Three: computing one input row at a time re-reads all of W for each row.
  // A block of 4 rows reads each weight vector once for all four, so a block
  // does 16 multiply-adds per 8 loads instead of 4 per 5. Sequences and
  // batches are where this pays.
  //
  // The kernels use explicit 4-wide vector types (Vec4). Written as plain
  // arrays, the compiler is free to vectorize across the four outputs instead
  // of along the row, and when W sits inline in the object it does exactly
  // that, with transposes and spills, at about half the speed.
  //
  // Summation order: per output, lanes over i, then (l0 + l1) + (l2 + l3),
  // then the leftover columns, then the bias -- the same for a block of rows
  // as for one row, so batching a row changes no result. Gradients also
  // accumulate row by row in order, so the 4-row backward matches the
  // one-row backward exactly.
  //
  // With bf16 storage the loads widen to float and every sum is in float; only
  // the stored outputs and dx are rounded to bf16.
  //
  // `benchmarks/tensor_bench.cpp` is where these claims are checked.
  static constexpr std::size_t kLanes = 4;
  static constexpr std::size_t kBlock = 4;     // outputs per block
  static constexpr std::size_t kRowBlock = 4;  // input rows per block
  static constexpr std::size_t kVecIn = InDim / kLanes * kLanes;
  static constexpr std::size_t kVecOut = OutDim / kBlock * kBlock;
  using V = Vec4<Acc>;

  // out[r * OutDim + o] = b + W x_r for R rows starting at x. `out` must not
  // overlap anything else. `w` is W itself or a float copy of it, and x is
  // T or float: the bf16 paths widen once and reuse the float kernel.
  template <std::size_t R, typename W, typename X>
  ZF_ALWAYS_INLINE void matmul_rows(const W* ZF_RESTRICT w,
                                    const X* ZF_RESTRICT x,
                                    T* ZF_RESTRICT out) const {
    for (std::size_t o = 0; o < kVecOut; o += kBlock) {
      V acc[R][kBlock] = {};
      for (std::size_t i = 0; i < kVecIn; i += kLanes) {
        V xv[R];
        for (std::size_t r = 0; r < R; ++r) xv[r] = load4(x + r * InDim + i);
        for (std::size_t c = 0; c < kBlock; ++c) {
          const V wv = load4(w + (o + c) * InDim + i);
          for (std::size_t r = 0; r < R; ++r) acc[r][c] += wv * xv[r];
        }
      }
      for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t c = 0; c < kBlock; ++c) {
          Acc sum = hsum4(acc[r][c]);
          // The widths are compile-time constants, so when they divide evenly
          // the leftover loops are not emitted at all.
          if constexpr (InDim % kLanes != 0) {
            const W* ZF_RESTRICT wo = w + (o + c) * InDim;
            for (std::size_t i = kVecIn; i < InDim; ++i) {
              sum += wo[i] * x[r * InDim + i];
            }
          }
          out[r * OutDim + o + c] = static_cast<T>(static_cast<Acc>(b_(o + c)) + sum);
        }
      }
    }
    if constexpr (OutDim % kBlock != 0) {
      for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t o = kVecOut; o < OutDim; ++o) {
          out[r * OutDim + o] = static_cast<T>(
              static_cast<Acc>(b_(o)) + dot_row(w + o * InDim, x + r * InDim));
        }
      }
    }
  }

  // For R rows: accumulates dW and db into dw, db, and adds each row's dx
  // into `dx`.
  // dx is built in a local block and added once at the end, so the inner
  // loop's read-modify-write stays in registers and L1.
  template <std::size_t R, typename W, typename X, typename G>
  ZF_ALWAYS_INLINE void matmul_rows_backward(const W* ZF_RESTRICT w,
                                             const X* ZF_RESTRICT x,
                                             const G* ZF_RESTRICT g,
                                             T* ZF_RESTRICT dx, Acc* ZF_RESTRICT dw,
                                             Acc* ZF_RESTRICT db) {
    alignas(16) Acc local[R][InDim] = {};
    for (std::size_t o = 0; o < kVecOut; o += kBlock) {
      Acc gr[R][kBlock];
      for (std::size_t c = 0; c < kBlock; ++c) {
        for (std::size_t r = 0; r < R; ++r) {
          gr[r][c] = g[r * OutDim + o + c];
          db[o + c] += gr[r][c];
        }
      }
      for (std::size_t i = 0; i < kVecIn; i += kLanes) {
        V xv[R];
        for (std::size_t r = 0; r < R; ++r) xv[r] = load4(x + r * InDim + i);
        V wv[kBlock];
        for (std::size_t c = 0; c < kBlock; ++c) {
          wv[c] = load4(w + (o + c) * InDim + i);
          V d = load4(dw + (o + c) * InDim + i);
          for (std::size_t r = 0; r < R; ++r) d += gr[r][c] * xv[r];
          store4(dw + (o + c) * InDim + i, d);
        }
        for (std::size_t r = 0; r < R; ++r) {
          V l = load4(&local[r][i]);
          l += gr[r][0] * wv[0] + gr[r][1] * wv[1] + gr[r][2] * wv[2] +
               gr[r][3] * wv[3];
          store4(&local[r][i], l);
        }
      }
      if constexpr (InDim % kLanes != 0) {
        for (std::size_t i = kVecIn; i < InDim; ++i) {
          for (std::size_t c = 0; c < kBlock; ++c) {
            for (std::size_t r = 0; r < R; ++r) {
              dw[(o + c) * InDim + i] += gr[r][c] * x[r * InDim + i];
            }
          }
          for (std::size_t r = 0; r < R; ++r) {
            local[r][i] += gr[r][0] * w[(o + 0) * InDim + i] +
                           gr[r][1] * w[(o + 1) * InDim + i] +
                           gr[r][2] * w[(o + 2) * InDim + i] +
                           gr[r][3] * w[(o + 3) * InDim + i];
          }
        }
      }
    }
    if constexpr (OutDim % kBlock != 0) {
      for (std::size_t o = kVecOut; o < OutDim; ++o) {
        Acc* ZF_RESTRICT dwo = dw + o * InDim;
        const W* ZF_RESTRICT wo = w + o * InDim;
        for (std::size_t r = 0; r < R; ++r) {
          const Acc gv = g[r * OutDim + o];
          db[o] += gv;
          for (std::size_t i = 0; i < InDim; ++i) {
            dwo[i] += gv * x[r * InDim + i];
            local[r][i] += gv * wo[i];
          }
        }
      }
    }
    for (std::size_t r = 0; r < R; ++r) {
      for (std::size_t i = 0; i < InDim; ++i) {
        dx[r * InDim + i] =
            static_cast<T>(static_cast<Acc>(dx[r * InDim + i]) + local[r][i]);
      }
    }
  }

  // Plain dot product over InDim with the kernels' summation order. Used for
  // the leftover outputs when OutDim is not a multiple of the block.
  template <typename A, typename B>
  static Acc dot_row(const A* ZF_RESTRICT a, const B* ZF_RESTRICT b) {
    V acc{};
    for (std::size_t i = 0; i < kVecIn; i += kLanes) acc += load4(a + i) * load4(b + i);
    Acc sum = hsum4(acc);
    for (std::size_t i = kVecIn; i < InDim; ++i) sum += a[i] * b[i];
    return sum;
  }

  // --- single vector ---

  Tensor<T, OutDim> forward(const Tensor<T, InDim>& x) const {
    Tensor<T, OutDim> out;
    matmul_rows<1>(w_.data(), x.data(), out.data());
    return out;
  }

  Tensor<T, InDim> backward(const Tensor<T, InDim>& x,
                            const Tensor<T, OutDim>& d_out) {
    Tensor<T, InDim> dx;
    matmul_rows_backward<1>(w_.data(), x.data(), d_out.data(), dx.data(), dw_.data(), db_.data());
    return dx;
  }

  // --- a stack of `Rows` vectors, e.g. a token sequence or a batch ---
  //
  // `rows` is how many of them are real. A padded sequence shares one
  // compile-time Rows; the rows past the valid length are read by nobody and
  // receive no gradient, so computing them is pure waste. Rows beyond the
  // limit are left zero.

  template <std::size_t Rows>
  Tensor<T, Rows, OutDim> forward(const Tensor<T, Rows, InDim>& x,
                                  std::size_t rows = Rows) const {
    Tensor<T, Rows, OutDim> out;
    forward_into(x.data(), out.data(), rows < Rows ? rows : Rows);
    return out;
  }

  template <std::size_t Rows>
  Tensor<T, Rows, InDim> backward(const Tensor<T, Rows, InDim>& x,
                                  const Tensor<T, Rows, OutDim>& d_out,
                                  std::size_t rows = Rows) {
    Tensor<T, Rows, InDim> dx;
    backward_into(x.data(), d_out.data(), dx.data(), rows < Rows ? rows : Rows);
    return dx;
  }

  // Raw-pointer forms: `rows` rows of x (InDim each) into out (OutDim each),
  // and the matching backward, which adds into dx.
  //
  // With the BLAS backend on, float stacks with enough work go to SGEMM
  // instead (blas::use_gemm; blas.h has the measurements). SGEMM sums in
  // its own order, so on that path a row's result can differ in the last bits
  // from the same row computed alone.
  //
  // bf16 stacks of at least kWidenMinRows rows widen W to float once per call,
  // and each block of rows once, and then run the float kernels (or SGEMM):
  // otherwise every weight would be widened again for every block of rows.
  // Widening is exact and the float kernels sum in the same order, so this
  // changes no result, only the time. Shorter stacks widen as they load.
  static constexpr bool kUseBlas = blas::kEnabled && std::is_same_v<T, float>;
  static constexpr bool kStorageType = !std::is_same_v<T, Acc>;
  static constexpr std::size_t kWidenMinRows = 8;

  void forward_into(const T* x, T* out, std::size_t rows) const {
    if constexpr (kUseBlas) {
      if (blas::use_gemm(rows, InDim, OutDim)) {
        blas::linear_forward(x, w_.data(), b_.data(), out, rows, InDim, OutDim);
        return;
      }
    }
    if (split_kernels(rows)) {
      parallel_for(&ThreadPool::global(), rows, kSplitRows,
                   [&](std::size_t, std::size_t r0, std::size_t r1) {
                     forward_kernels(x + r0 * InDim, out + r0 * OutDim, r1 - r0);
                   });
      return;
    }
    forward_kernels(x, out, rows);
  }

  // Without the GEMM, a large stack splits its rows over
  // ThreadPool::global() in chunks of kSplitRows: forward is the same row by
  // row; backward sums each chunk's dW and db apart and adds them in chunk
  // order, so the result depends on the rows, not on the threads.
  static constexpr std::size_t kSplitRows = 64;
  static bool split_kernels(std::size_t rows) {
    return rows > kSplitRows && rows * InDim * OutDim >= blas::kSplitWork &&
           !ThreadPool::on_worker() && ThreadPool::global().participants() > 1;
  }

  void forward_kernels(const T* x, T* out, std::size_t rows) const {
    if constexpr (kStorageType) {
      if (rows >= kWidenMinRows) {
        forward_widened(x, out, rows);
        return;
      }
    }
    std::size_t r = 0;
    for (; r + kRowBlock <= rows; r += kRowBlock) {
      matmul_rows<kRowBlock>(w_.data(), x + r * InDim, out + r * OutDim);
    }
    for (; r < rows; ++r) matmul_rows<1>(w_.data(), x + r * InDim, out + r * OutDim);
  }

  // Span forms over a row-stacked tensor: only rows inside `spans` are
  // computed. forward_spans overwrites those rows of `out` and leaves the
  // rest alone; backward_spans adds into those rows of `dx`.
  //
  // Spans that touch are merged into one call: a batch of full-length
  // examples is then a single tall matmul instead of one per example.
  template <std::size_t Rows>
  void forward_spans(const Tensor<T, Rows, InDim>& x, Tensor<T, Rows, OutDim>& out,
                     RowSpans spans) const {
    for_merged_spans(spans, [&](std::size_t start, std::size_t count) {
      forward_into(x.data() + start * InDim, out.data() + start * OutDim, count);
    });
  }

  template <std::size_t Rows>
  void backward_spans(const Tensor<T, Rows, InDim>& x,
                      const Tensor<T, Rows, OutDim>& d_out,
                      Tensor<T, Rows, InDim>& dx, RowSpans spans) {
    for_merged_spans(spans, [&](std::size_t start, std::size_t count) {
      backward_into(x.data() + start * InDim, d_out.data() + start * OutDim,
                    dx.data() + start * InDim, count);
    });
  }

  template <typename Body>
  static void for_merged_spans(RowSpans spans, Body&& body) {
    std::size_t start = 0;
    std::size_t count = 0;
    for (const RowSpan& s : spans) {
      if (s.count == 0) continue;
      if (count != 0 && s.start == start + count) {
        count += s.count;
        continue;
      }
      if (count != 0) body(start, count);
      start = s.start;
      count = s.count;
    }
    if (count != 0) body(start, count);
  }

  void backward_into(const T* x, const T* d_out, T* dx, std::size_t rows) {
    if constexpr (kUseBlas) {
      if (blas::use_gemm(rows, InDim, OutDim)) {
        blas::linear_backward(x, d_out, w_.data(), dx, dw_.data(), db_.data(),
                              rows, InDim, OutDim);
        return;
      }
    }
    if (split_kernels(rows)) {
      const std::size_t chunks = chunk_count(rows, kSplitRows);
      constexpr std::size_t kPart = OutDim * InDim + OutDim;
      std::vector<Acc> part(chunks * kPart, Acc{});
      parallel_for(&ThreadPool::global(), rows, kSplitRows,
                   [&](std::size_t c, std::size_t r0, std::size_t r1) {
                     Acc* dw = part.data() + c * kPart;
                     backward_kernels(x + r0 * InDim, d_out + r0 * OutDim, dx + r0 * InDim,
                                      r1 - r0, dw, dw + OutDim * InDim);
                   });
      for (std::size_t c = 0; c < chunks; ++c) {
        const Acc* p = part.data() + c * kPart;
        for (std::size_t i = 0; i < OutDim * InDim; ++i) dw_.data()[i] += p[i];
        for (std::size_t o = 0; o < OutDim; ++o) db_(o) += p[OutDim * InDim + o];
      }
      return;
    }
    backward_kernels(x, d_out, dx, rows, dw_.data(), db_.data());
  }

  void backward_kernels(const T* x, const T* d_out, T* dx, std::size_t rows, Acc* dw,
                        Acc* db) {
    if constexpr (kStorageType) {
      if (rows >= kWidenMinRows) {
        backward_widened(x, d_out, dx, rows, dw, db);
        return;
      }
    }
    std::size_t r = 0;
    for (; r + kRowBlock <= rows; r += kRowBlock) {
      matmul_rows_backward<kRowBlock>(w_.data(), x + r * InDim, d_out + r * OutDim,
                                      dx + r * InDim, dw, db);
    }
    for (; r < rows; ++r) {
      matmul_rows_backward<1>(w_.data(), x + r * InDim, d_out + r * OutDim,
                              dx + r * InDim, dw, db);
    }
  }

  // The bf16 paths described at forward_into. Scratch is per thread and
  // reused, so a call allocates nothing once warmed up.
  void forward_widened(const T* x, T* out, std::size_t rows) const {
    Acc* wf = detail::widen_scratch<Acc>(0, OutDim * InDim);
    widen_into(w_.data(), wf, OutDim * InDim);
    if constexpr (blas::kEnabled) {
      if (blas::use_gemm(rows, InDim, OutDim)) {
        Acc* xf = detail::widen_scratch<Acc>(1, rows * InDim);
        Acc* bf = detail::widen_scratch<Acc>(2, OutDim);
        Acc* of = detail::widen_scratch<Acc>(3, rows * OutDim);
        widen_into(x, xf, rows * InDim);
        widen_into(b_.data(), bf, OutDim);
        blas::linear_forward(xf, wf, bf, of, rows, InDim, OutDim);
        narrow_into(of, out, rows * OutDim);
        return;
      }
    }
    Acc* xb = detail::widen_scratch<Acc>(1, kRowBlock * InDim);
    std::size_t r = 0;
    for (; r + kRowBlock <= rows; r += kRowBlock) {
      widen_into(x + r * InDim, xb, kRowBlock * InDim);
      matmul_rows<kRowBlock>(wf, xb, out + r * OutDim);
    }
    for (; r < rows; ++r) matmul_rows<1>(wf, x + r * InDim, out + r * OutDim);
  }

  void backward_widened(const T* x, const T* d_out, T* dx, std::size_t rows, Acc* dw,
                        Acc* db) {
    Acc* wf = detail::widen_scratch<Acc>(0, OutDim * InDim);
    widen_into(w_.data(), wf, OutDim * InDim);
    if constexpr (blas::kEnabled) {
      if (blas::use_gemm(rows, InDim, OutDim)) {
        Acc* xf = detail::widen_scratch<Acc>(1, rows * InDim);
        Acc* gf = detail::widen_scratch<Acc>(2, rows * OutDim);
        Acc* dxf = detail::widen_scratch<Acc>(3, rows * InDim);
        widen_into(x, xf, rows * InDim);
        widen_into(d_out, gf, rows * OutDim);
        widen_into(dx, dxf, rows * InDim);
        blas::linear_backward(xf, gf, wf, dxf, dw, db, rows, InDim, OutDim);
        narrow_into(dxf, dx, rows * InDim);
        return;
      }
    }
    Acc* xb = detail::widen_scratch<Acc>(1, kRowBlock * InDim);
    Acc* gb = detail::widen_scratch<Acc>(2, kRowBlock * OutDim);
    std::size_t r = 0;
    for (; r + kRowBlock <= rows; r += kRowBlock) {
      widen_into(x + r * InDim, xb, kRowBlock * InDim);
      widen_into(d_out + r * OutDim, gb, kRowBlock * OutDim);
      matmul_rows_backward<kRowBlock>(wf, xb, gb, dx + r * InDim, dw, db);
    }
    for (; r < rows; ++r) {
      matmul_rows_backward<1>(wf, x + r * InDim, d_out + r * OutDim, dx + r * InDim, dw, db);
    }
  }

  void zero_grad() {
    dw_.fill(Acc{});
    db_.fill(Acc{});
  }

  void apply_sgd(Acc lr) {
    for (std::size_t i = 0; i < w_.size(); ++i) {
      w_[i] = static_cast<T>(static_cast<Acc>(w_[i]) - lr * dw_[i]);
    }
    for (std::size_t i = 0; i < b_.size(); ++i) {
      b_[i] = static_cast<T>(static_cast<Acc>(b_[i]) - lr * db_[i]);
    }
  }

  WeightTensor& weight() { return w_; }
  const WeightTensor& weight() const { return w_; }
  BiasTensor& bias() { return b_; }
  const BiasTensor& bias() const { return b_; }
  const WeightGradTensor& weight_grad() const { return dw_; }
  const BiasGradTensor& bias_grad() const { return db_; }

 private:
  WeightTensor w_;
  BiasTensor b_;
  WeightGradTensor dw_;
  BiasGradTensor db_;
};

// Normalizes each row over its `Dim` features, then scales and shifts.
template <typename T, std::size_t Dim>
class LayerNorm {
  static_assert(is_real_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  using Acc = accumulator_t<T>;  // statistics and gradients
  static constexpr std::size_t dim = Dim;
  static constexpr std::size_t parameter_count = 2 * Dim;

  explicit LayerNorm(Acc eps = static_cast<Acc>(1e-5)) : eps_(eps) {
    gamma_.fill(T{1});
  }

  Tensor<T, Dim> forward(const Tensor<T, Dim>& x) const {
    Tensor<T, Dim> out;
    normalize_row(x.data(), out.data());
    return out;
  }

  Tensor<T, Dim> backward(const Tensor<T, Dim>& x,
                          const Tensor<T, Dim>& d_out) {
    Tensor<T, Dim> dx;
    normalize_row_backward(x.data(), d_out.data(), dx.data());
    return dx;
  }

  // Row-wise over a stack of vectors.
  //
  // Normalizes `rows` rows in place rather than copying each one into a
  // temporary, through the single-vector path, and back out again: that was
  // three heap-allocated tensors per row, which for a 48-row sequence came to
  // 144 allocations to do 48 rows of arithmetic.
  //
  // `rows` is how many rows are real; the rest are left zero.
  template <std::size_t Rows>
  Tensor<T, Rows, Dim> forward(const Tensor<T, Rows, Dim>& x,
                               std::size_t rows = Rows) const {
    Tensor<T, Rows, Dim> out;
    const std::size_t limit = rows < Rows ? rows : Rows;
    const T* ZF_RESTRICT xp = x.data();
    T* ZF_RESTRICT op = out.data();
    for (std::size_t r = 0; r < limit; ++r) {
      normalize_row(xp + r * Dim, op + r * Dim);
    }
    return out;
  }

  template <std::size_t Rows>
  Tensor<T, Rows, Dim> backward(const Tensor<T, Rows, Dim>& x,
                                const Tensor<T, Rows, Dim>& d_out,
                                std::size_t rows = Rows) {
    Tensor<T, Rows, Dim> dx;
    const std::size_t limit = rows < Rows ? rows : Rows;
    const T* ZF_RESTRICT xp = x.data();
    const T* ZF_RESTRICT gp = d_out.data();
    T* ZF_RESTRICT dxp = dx.data();
    for (std::size_t r = 0; r < limit; ++r) {
      normalize_row_backward(xp + r * Dim, gp + r * Dim, dxp + r * Dim);
    }
    return dx;
  }

  // Only the rows inside `spans`; the rest of the result is zero.
  template <std::size_t Rows>
  Tensor<T, Rows, Dim> forward(const Tensor<T, Rows, Dim>& x, RowSpans spans) const {
    Tensor<T, Rows, Dim> out;
    for (const RowSpan& s : spans) {
      for (std::size_t r = s.start; r < s.start + s.count; ++r) {
        normalize_row(x.data() + r * Dim, out.data() + r * Dim);
      }
    }
    return out;
  }

  template <std::size_t Rows>
  Tensor<T, Rows, Dim> backward(const Tensor<T, Rows, Dim>& x,
                                const Tensor<T, Rows, Dim>& d_out, RowSpans spans) {
    Tensor<T, Rows, Dim> dx;
    for (const RowSpan& s : spans) {
      for (std::size_t r = s.start; r < s.start + s.count; ++r) {
        normalize_row_backward(x.data() + r * Dim, d_out.data() + r * Dim,
                               dx.data() + r * Dim);
      }
    }
    return dx;
  }

  // One row, in place; the single-vector path is the same call.
  //
  // The row's reductions (the mean, the variance, and in backward the two
  // sums normalization couples in) run in four 4-wide accumulators, so
  // sixteen independent chains instead of one: a single chain waits on each
  // add's latency, which for a wide row was most of the cost. Summation
  // order: blocks of 16 into the accumulators, (a0 + a1) + (a2 + a3), the
  // leftover 4-blocks into that, its lanes (l0 + l1) + (l2 + l3), then the
  // leftover elements. The elementwise passes need no help.
  ZF_ALWAYS_INLINE void normalize_row(const T* ZF_RESTRICT x,
                                      T* ZF_RESTRICT out) const {
    Acc mean, rstd;
    row_stats(x, &mean, &rstd);
    const T* ZF_RESTRICT g = gamma_.data();
    const T* ZF_RESTRICT b = beta_.data();
    for (std::size_t i = 0; i < Dim; ++i) {
      out[i] = static_cast<T>(g[i] * (x[i] - mean) * rstd + b[i]);
    }
  }

  ZF_ALWAYS_INLINE void normalize_row_backward(const T* ZF_RESTRICT x,
                                              const T* ZF_RESTRICT d_out,
                                              T* ZF_RESTRICT dx) {
    row_backward(x, d_out, dx, dgamma_.data(), dbeta_.data());
  }

  // A stack of `rows` rows (raw, Dim each), split over `pool` in chunks of
  // kRowGrain rows; nullptr runs them here. forward_rows overwrites out.
  void forward_rows(const T* x, T* out, std::size_t rows, ThreadPool* pool = nullptr) const {
    parallel_for(pool, rows, kRowGrain, [&](std::size_t, std::size_t r0, std::size_t r1) {
      for (std::size_t r = r0; r < r1; ++r) normalize_row(x + r * Dim, out + r * Dim);
    });
  }

  // Overwrites dx; adds into the parameter gradients. Each chunk sums its
  // own dgamma, dbeta, added in chunk order afterwards, so the result is the
  // same on any pool (the chunks depend only on `rows`).
  void backward_rows(const T* x, const T* d_out, T* dx, std::size_t rows,
                     ThreadPool* pool = nullptr) {
    const std::size_t chunks = chunk_count(rows, kRowGrain);
    if (chunks <= 1) {
      for (std::size_t r = 0; r < rows; ++r) {
        normalize_row_backward(x + r * Dim, d_out + r * Dim, dx + r * Dim);
      }
      return;
    }
    std::vector<Acc> part(chunks * 2 * Dim, Acc{});
    parallel_for(pool, rows, kRowGrain, [&](std::size_t c, std::size_t r0, std::size_t r1) {
      Acc* dg = part.data() + c * 2 * Dim;
      for (std::size_t r = r0; r < r1; ++r) {
        row_backward(x + r * Dim, d_out + r * Dim, dx + r * Dim, dg, dg + Dim);
      }
    });
    for (std::size_t c = 0; c < chunks; ++c) {
      const Acc* dg = part.data() + c * 2 * Dim;
      for (std::size_t i = 0; i < Dim; ++i) {
        dgamma_(i) += dg[i];
        dbeta_(i) += dg[Dim + i];
      }
    }
  }

  // Rows per chunk when a stack is split over threads.
  static constexpr std::size_t kRowGrain = 32;

 private:
  // One row's backward, its dgamma and dbeta added into dg, db.
  ZF_ALWAYS_INLINE void row_backward(const T* ZF_RESTRICT x, const T* ZF_RESTRICT d_out,
                                     T* ZF_RESTRICT dx, Acc* ZF_RESTRICT dg,
                                     Acc* ZF_RESTRICT db) {
    Acc mean, rstd;
    row_stats(x, &mean, &rstd);
    const T* ZF_RESTRICT g = gamma_.data();
    for (std::size_t i = 0; i < Dim; ++i) {
      const Acc xhat = (x[i] - mean) * rstd;
      dg[i] += d_out[i] * xhat;
      db[i] += d_out[i];
    }
    // dy scaled by gamma, plus the two mean terms normalization couples in
    const V neg_mean = splat(-mean), rstd_v = splat(rstd);
    const auto dgi = [&](std::size_t i) { return load4(d_out + i) * load4(g + i); };
    const auto xhat = [&](std::size_t i) { return (load4(x + i) + neg_mean) * rstd_v; };
    Acc sum_dg = lane_sum(dgi);
    Acc sum_dg_xhat = lane_sum([&](std::size_t i) { return dgi(i) * xhat(i); });
    for (std::size_t i = kVec; i < Dim; ++i) {
      const Acc dgi = d_out[i] * g[i];
      sum_dg += dgi;
      sum_dg_xhat += dgi * ((x[i] - mean) * rstd);
    }
    const Acc inv_n = Acc{1} / static_cast<Acc>(Dim);
    for (std::size_t i = 0; i < Dim; ++i) {
      const Acc xh = (x[i] - mean) * rstd;
      const Acc dg_i = d_out[i] * g[i];
      dx[i] = static_cast<T>(rstd * (dg_i - inv_n * sum_dg - xh * inv_n * sum_dg_xhat));
    }
  }

 public:
  void zero_grad() {
    dgamma_.fill(Acc{});
    dbeta_.fill(Acc{});
  }

  void apply_sgd(Acc lr) {
    for (std::size_t i = 0; i < Dim; ++i) {
      gamma_(i) = static_cast<T>(static_cast<Acc>(gamma_(i)) - lr * dgamma_(i));
      beta_(i) = static_cast<T>(static_cast<Acc>(beta_(i)) - lr * dbeta_(i));
    }
  }

  Tensor<T, Dim>& gamma() { return gamma_; }
  const Tensor<T, Dim>& gamma() const { return gamma_; }
  Tensor<T, Dim>& beta() { return beta_; }
  const Tensor<T, Dim>& beta() const { return beta_; }
  const Tensor<Acc, Dim>& gamma_grad() const { return dgamma_; }
  const Tensor<Acc, Dim>& beta_grad() const { return dbeta_; }

 private:
  static constexpr std::size_t kLanes = 4;
  static constexpr std::size_t kVec = Dim / kLanes * kLanes;
  static constexpr std::size_t kBlock = 4 * kLanes;
  static constexpr std::size_t kVecBlock = Dim / kBlock * kBlock;
  using V = Vec4<Acc>;

  // The sum over i < kVec of term(i), a V of the four elements at i, in the
  // order described at normalize_row. The elements past kVec are the
  // caller's.
  template <typename Term>
  static ZF_ALWAYS_INLINE Acc lane_sum(Term&& term) {
    V a0{}, a1{}, a2{}, a3{};
    for (std::size_t i = 0; i < kVecBlock; i += kBlock) {
      a0 += term(i);
      a1 += term(i + kLanes);
      a2 += term(i + 2 * kLanes);
      a3 += term(i + 3 * kLanes);
    }
    V a = (a0 + a1) + (a2 + a3);
    for (std::size_t i = kVecBlock; i < kVec; i += kLanes) a += term(i);
    return hsum4(a);
  }

  static ZF_ALWAYS_INLINE V splat(Acc a) {
    V v{};
    for (std::size_t k = 0; k < kLanes; ++k) v[k] = a;
    return v;
  }

  // A row's mean and 1 / sqrt(var + eps).
  ZF_ALWAYS_INLINE void row_stats(const T* ZF_RESTRICT x, Acc* mean,
                                  Acc* rstd) const {
    Acc sum = lane_sum([&](std::size_t i) { return load4(x + i); });
    for (std::size_t i = kVec; i < Dim; ++i) sum += x[i];
    const Acc m = sum / static_cast<Acc>(Dim);
    const V neg_m = splat(-m);
    Acc var = lane_sum([&](std::size_t i) {
      const V c = load4(x + i) + neg_m;
      return c * c;
    });
    for (std::size_t i = kVec; i < Dim; ++i) {
      const Acc c = x[i] - m;
      var += c * c;
    }
    var /= static_cast<Acc>(Dim);
    *mean = m;
    *rstd = Acc{1} / std::sqrt(var + eps_);
  }

  template <std::size_t Rows>
  static Tensor<T, Dim> extract_row(const Tensor<T, Rows, Dim>& m,
                                    std::size_t r) {
    Tensor<T, Dim> row;
    for (std::size_t i = 0; i < Dim; ++i) {
      row(i) = m(r, i);
    }
    return row;
  }

  Tensor<T, Dim> gamma_;
  Tensor<T, Dim> beta_;
  Tensor<Acc, Dim> dgamma_;
  Tensor<Acc, Dim> dbeta_;
  Acc eps_;
};

// Token id -> dense row. Gradients are scattered back per id.
template <typename T, std::size_t VocabSize, std::size_t EmbedDim>
class Embedding {
  static_assert(is_real_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  using Acc = accumulator_t<T>;
  using WeightTensor = Tensor<T, VocabSize, EmbedDim>;
  using WeightGradTensor = Tensor<Acc, VocabSize, EmbedDim>;

  static constexpr std::size_t vocab_size = VocabSize;
  static constexpr std::size_t embed_dim = EmbedDim;
  static constexpr std::size_t parameter_count = VocabSize * EmbedDim;

  Embedding() = default;

  template <typename Rng>
  void randomize(Rng& rng, T stddev = static_cast<T>(0.02)) {
    std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));
    for (T& v : w_) {
      v = static_cast<T>(dist(rng));
    }
  }

  // Throws std::out_of_range when `id` is outside the vocabulary.
  Tensor<T, EmbedDim> forward(std::size_t id) const {
    Tensor<T, EmbedDim> out;
    for (std::size_t i = 0; i < EmbedDim; ++i) {
      out(i) = w_.at(id, i);
    }
    return out;
  }

  // Accumulates dL/dw for the row `id`.
  void backward(std::size_t id, const Tensor<T, EmbedDim>& d_out) {
    for (std::size_t i = 0; i < EmbedDim; ++i) {
      dw_.at(id, i) += d_out(i);
    }
  }

  void zero_grad() { dw_.fill(Acc{}); }

  void apply_sgd(Acc lr) {
    for (std::size_t i = 0; i < w_.size(); ++i) {
      w_[i] = static_cast<T>(static_cast<Acc>(w_[i]) - lr * dw_[i]);
    }
  }

  WeightTensor& weight() { return w_; }
  const WeightTensor& weight() const { return w_; }
  const WeightGradTensor& weight_grad() const { return dw_; }

 private:
  WeightTensor w_;
  WeightGradTensor dw_;
};

// In-place softmax over `n` leading entries of `row`, max-shifted.
// exp for the softmax, restricted to the non-positive arguments a softmax
// produces after its maximum has been subtracted.
//
// Why not std::exp: it is a library call, so the loop around it cannot be
// vectorized and each element costs a call. In the attention kernels the
// softmax was the single largest item -- more than the scores and the context
// put together -- entirely because of that.
//
// The method is the standard range reduction. Write x = k*ln2 + r with k an
// integer and |r| <= ln2/2, so exp(x) = 2^k * exp(r); 2^k is built by placing
// k in the exponent field of a float, and exp(r) over that narrow interval is
// a short polynomial -- the Taylor series through r^6/720.
//
// Measured against std::exp over the whole range, the worst relative error is
// 4.3e-7, which is three or four units in the last place of a float. Most of
// that is rounding in the evaluation rather than truncation of the series:
// stopping at r^5/120 gives 3.3e-6, and reaching one ulp would need the range
// reduction done in two pieces to keep r itself exact. Three ulp is far below
// anything a softmax over attention scores can notice, and `SoftmaxExp` in the
// test suite measures it rather than taking it on trust.
inline float softmax_exp(float x) {
  constexpr float kLog2e = 1.44269504088896341f;
  constexpr float kLn2 = 0.69314718055994531f;
  // The smallest x whose result is still a normal float: below it 2^k is not
  // representable and zero is the right answer. The limit is derived from the
  // exponent range rather than guessed -- an earlier cutoff at -87.3 threw
  // away values near 1.2e-38, which are normal floats.
  constexpr float kUnderflow = -87.33654f;  // -126 * ln2
  // Written as selects rather than early returns: a branch in here stops the
  // enclosing softmax loop from vectorizing, which was the whole reason for
  // not calling std::exp.
  //
  // The clamp is std::fmax because it returns the other operand for a NaN,
  // which keeps the value fed to the integer conversion below finite -- a NaN
  // converted to int32 is undefined behaviour, and this is a hot path, not a
  // place to leave that lying around. A NaN still comes out of the function,
  // through the (x - x) term: it is zero for every finite input and NaN for a
  // NaN, so a NaN propagates the way it does through std::exp instead of
  // quietly becoming a weight of zero. Attention scores going NaN is a bug
  // upstream, and hiding it here would only move the symptom.
  const bool under = x < kUnderflow;  // false for a NaN
  const float xc = std::fmax(x, kUnderflow);
  const float kf = std::nearbyint(xc * kLog2e);
  const auto k = static_cast<std::int32_t>(kf);
  const float r = xc - kf * kLn2;
  // exp(r) on |r| <= ln2/2
  const float p =
      1.0f +
      r * (1.0f +
           r * (0.5f +
                r * (0.16666667f +
                     r * (0.041666668f +
                          r * (0.008333334f + r * 0.0013888889f)))));
  // 2^k by exponent assembly: (k + 127) << 23 is the bit pattern of 2^k.
  const std::uint32_t bits = static_cast<std::uint32_t>(k + 127) << 23;
  float two_k;
  std::memcpy(&two_k, &bits, sizeof(two_k));
  return under ? 0.0f : p * two_k + (x - x);
}

// Any other element type falls back to the library, which keeps the template
// honest for a double-precision instantiation.
template <typename T>
inline T softmax_exp_for(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return softmax_exp(x);
  } else {
    return std::exp(x);
  }
}

template <typename T>
void softmax_inplace(T* ZF_RESTRICT row, std::size_t n) {
  if constexpr (!std::is_same_v<T, accumulator_t<T>>) {
    // A storage type (bf16): rounding each exponential to 16 bits before
    // summing would put that error into every weight. Sum in float, then
    // exponentiate again for the stores -- the row has no float scratch.
    using Acc = accumulator_t<T>;
    Acc max_v = row[0];
    for (std::size_t i = 1; i < n; ++i) max_v = row[i] > max_v ? Acc(row[i]) : max_v;
    Acc sum{};
    for (std::size_t i = 0; i < n; ++i) sum += softmax_exp_for(Acc(row[i]) - max_v);
    const Acc inv = Acc{1} / sum;
    for (std::size_t i = 0; i < n; ++i) {
      row[i] = static_cast<T>(softmax_exp_for(Acc(row[i]) - max_v) * inv);
    }
  } else {
    T max_v = row[0];
    for (std::size_t i = 1; i < n; ++i) {
      if (row[i] > max_v) {
        max_v = row[i];
      }
    }
    // Exponentiate first, total afterwards. Accumulating the sum in the same
    // loop as the stores puts a dependent add between consecutive elements and
    // the loop will not vectorize; separated, the first loop is pure elementwise
    // work and the second is a reduction with four independent accumulators.
    for (std::size_t i = 0; i < n; ++i) {
      row[i] = softmax_exp_for(row[i] - max_v);
    }
    T acc[4] = {};
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      for (std::size_t k = 0; k < 4; ++k) {
        acc[k] += row[i + k];
      }
    }
    T sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (; i < n; ++i) {
      sum += row[i];
    }
    const T inv = T{1} / sum;
    for (std::size_t j = 0; j < n; ++j) {
      row[j] *= inv;
    }
  }
}

}  // namespace zf
