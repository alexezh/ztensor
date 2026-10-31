#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <random>
#include <type_traits>

#include "tensor.h"

// Primitive layers shared by the models.
//
// Convention: primitives are stateless with respect to activations. `forward`
// is const, and `backward(x, dOut)` is handed the same input `x` that produced
// the output; it accumulates parameter gradients and returns dL/dx. The
// composite models (Mlp, TransformerBlock, QueryModel, ...) are the ones that
// cache activations, because only they know every intermediate shape.

namespace zf {

enum class Activation { kIdentity, kRelu, kGelu };

template <typename T>
T activation_apply(Activation act, T x) {
  switch (act) {
    case Activation::kRelu:
      return x > T{0} ? x : T{0};
    case Activation::kGelu: {
      // tanh approximation of GELU
      const T c = static_cast<T>(0.7978845608028654);  // sqrt(2/pi)
      const T u = c * (x + static_cast<T>(0.044715) * x * x * x);
      return static_cast<T>(0.5) * x * (T{1} + std::tanh(u));
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
      const T t = std::tanh(u);
      const T du = c * (T{1} + static_cast<T>(3 * 0.044715) * x2);
      return static_cast<T>(0.5) * (T{1} + t) +
             static_cast<T>(0.5) * x * (T{1} - t * t) * du;
    }
    case Activation::kIdentity:
      break;
  }
  return T{1};
}

// y = W x + b, with W stored as [OutDim, InDim].
template <typename T, std::size_t InDim, std::size_t OutDim>
class Linear {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  using WeightTensor = Tensor<T, OutDim, InDim>;
  using BiasTensor = Tensor<T, OutDim>;

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
  // These two do all the arithmetic in the library: every projection, every
  // feed-forward layer and every head is a call into one of them. They are
  // written for the machine rather than for the page, so the shape is
  // explained here.
  //
  // Two things cost time in the obvious triple loop.
  //
  // One: `sum += w[i] * x[i]` is a chain where every multiply-add waits for
  // the one before it, so the loop runs at the latency of an FMA instead of
  // its throughput. Reassociating floating-point addition changes the result,
  // so a compiler will not split the chain unless told it may. Four partial
  // sums are four independent chains, and also the shape a 4-wide vector
  // reduction wants.
  //
  // Two: computing one output at a time re-reads the whole input row for each
  // row of W. Four outputs at once read the input row once and keep sixteen
  // accumulator lanes in flight, which is enough independent work to fill the
  // pipeline.
  //
  // Together these are about five times faster than the straightforward
  // version on the shapes this library uses. The summation order differs from
  // strict left-to-right; measured against single-accumulator summation the
  // worst relative difference is around 6e-7, and every backward() here is
  // checked against central differences, which is a far coarser instrument.
  //
  // `benchmarks/tensor_bench.cpp` is where these claims are checked.
  static constexpr std::size_t kLanes = 4;
  static constexpr std::size_t kBlock = 4;

  // Plain dot product with four independent accumulators. Used for the
  // leftover outputs when OutDim is not a multiple of the block.
  static T dot_row(const T* ZF_RESTRICT a, const T* ZF_RESTRICT b,
                   std::size_t n) {
    T acc[kLanes] = {};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
      for (std::size_t k = 0; k < kLanes; ++k) {
        acc[k] += a[i + k] * b[i + k];
      }
    }
    T sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (; i < n; ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  // out[0..OutDim) = b + W x, for one input row. `out` must not overlap
  // anything else.
  ZF_ALWAYS_INLINE void matvec(const T* ZF_RESTRICT x,
                               T* ZF_RESTRICT out) const {
    const T* ZF_RESTRICT w = w_.data();
    std::size_t o = 0;
    for (; o + kBlock <= OutDim; o += kBlock) {
      const T* ZF_RESTRICT w0 = w + (o + 0) * InDim;
      const T* ZF_RESTRICT w1 = w + (o + 1) * InDim;
      const T* ZF_RESTRICT w2 = w + (o + 2) * InDim;
      const T* ZF_RESTRICT w3 = w + (o + 3) * InDim;
      T a0[kLanes] = {};
      T a1[kLanes] = {};
      T a2[kLanes] = {};
      T a3[kLanes] = {};
      std::size_t i = 0;
      for (; i + kLanes <= InDim; i += kLanes) {
        for (std::size_t k = 0; k < kLanes; ++k) {
          const T xv = x[i + k];
          a0[k] += w0[i + k] * xv;
          a1[k] += w1[i + k] * xv;
          a2[k] += w2[i + k] * xv;
          a3[k] += w3[i + k] * xv;
        }
      }
      T s0 = (a0[0] + a0[1]) + (a0[2] + a0[3]);
      T s1 = (a1[0] + a1[1]) + (a1[2] + a1[3]);
      T s2 = (a2[0] + a2[1]) + (a2[2] + a2[3]);
      T s3 = (a3[0] + a3[1]) + (a3[2] + a3[3]);
      // The widths are compile-time constants, so when they divide evenly the
      // leftover loops are not merely unexecuted, they are not emitted. Left
      // in as runtime code they cost more than the arithmetic they guard: the
      // compiler cannot prove them dead, so it stops unrolling the loop above
      // and spills the accumulators.
      if constexpr (InDim % kLanes != 0) {
        for (; i < InDim; ++i) {
          const T xv = x[i];
          s0 += w0[i] * xv;
          s1 += w1[i] * xv;
          s2 += w2[i] * xv;
          s3 += w3[i] * xv;
        }
      }
      out[o + 0] = b_(o + 0) + s0;
      out[o + 1] = b_(o + 1) + s1;
      out[o + 2] = b_(o + 2) + s2;
      out[o + 3] = b_(o + 3) + s3;
    }
    if constexpr (OutDim % kBlock != 0) {
      for (; o < OutDim; ++o) {
        out[o] = b_(o) + dot_row(w + o * InDim, x, InDim);
      }
    }
  }

  // Accumulates dW and db for one row, and adds that row's dx into `dx`.
  //
  // dx is built up in a local row and added once at the end: accumulating
  // straight into the output would read and write the same memory OutDim
  // times, where a local row stays in registers and cache.
  ZF_ALWAYS_INLINE void matvec_backward(const T* ZF_RESTRICT x,
                                        const T* ZF_RESTRICT g,
                                        T* ZF_RESTRICT dx) {
    const T* ZF_RESTRICT w = w_.data();
    T* ZF_RESTRICT dw = dw_.data();
    T local[InDim] = {};
    std::size_t o = 0;
    for (; o + kBlock <= OutDim; o += kBlock) {
      const T g0 = g[o + 0];
      const T g1 = g[o + 1];
      const T g2 = g[o + 2];
      const T g3 = g[o + 3];
      db_(o + 0) += g0;
      db_(o + 1) += g1;
      db_(o + 2) += g2;
      db_(o + 3) += g3;
      T* ZF_RESTRICT d0 = dw + (o + 0) * InDim;
      T* ZF_RESTRICT d1 = dw + (o + 1) * InDim;
      T* ZF_RESTRICT d2 = dw + (o + 2) * InDim;
      T* ZF_RESTRICT d3 = dw + (o + 3) * InDim;
      const T* ZF_RESTRICT w0 = w + (o + 0) * InDim;
      const T* ZF_RESTRICT w1 = w + (o + 1) * InDim;
      const T* ZF_RESTRICT w2 = w + (o + 2) * InDim;
      const T* ZF_RESTRICT w3 = w + (o + 3) * InDim;
      for (std::size_t i = 0; i < InDim; ++i) {
        const T xv = x[i];
        d0[i] += g0 * xv;
        d1[i] += g1 * xv;
        d2[i] += g2 * xv;
        d3[i] += g3 * xv;
        local[i] += g0 * w0[i] + g1 * w1[i] + g2 * w2[i] + g3 * w3[i];
      }
    }
    if constexpr (OutDim % kBlock != 0) {
      for (; o < OutDim; ++o) {
        const T gv = g[o];
        db_(o) += gv;
        T* ZF_RESTRICT dwo = dw + o * InDim;
        const T* ZF_RESTRICT wo = w + o * InDim;
        for (std::size_t i = 0; i < InDim; ++i) {
          dwo[i] += gv * x[i];
          local[i] += gv * wo[i];
        }
      }
    }
    for (std::size_t i = 0; i < InDim; ++i) {
      dx[i] += local[i];
    }
  }

  // --- single vector ---

  Tensor<T, OutDim> forward(const Tensor<T, InDim>& x) const {
    Tensor<T, OutDim> out;
    matvec(x.data(), out.data());
    return out;
  }

  Tensor<T, InDim> backward(const Tensor<T, InDim>& x,
                            const Tensor<T, OutDim>& d_out) {
    Tensor<T, InDim> dx;
    matvec_backward(x.data(), d_out.data(), dx.data());
    return dx;
  }

  // --- a stack of `Rows` vectors, e.g. a token sequence ---
  //
  // `rows` is how many of them are real. A padded sequence shares one
  // compile-time Rows; the rows past the valid length are read by nobody and
  // receive no gradient, so computing them is pure waste. Rows beyond the
  // limit are left zero.

  template <std::size_t Rows>
  Tensor<T, Rows, OutDim> forward(const Tensor<T, Rows, InDim>& x,
                                  std::size_t rows = Rows) const {
    Tensor<T, Rows, OutDim> out;
    const std::size_t limit = rows < Rows ? rows : Rows;
    for (std::size_t r = 0; r < limit; ++r) {
      matvec(x.data() + r * InDim, out.data() + r * OutDim);
    }
    return out;
  }

  template <std::size_t Rows>
  Tensor<T, Rows, InDim> backward(const Tensor<T, Rows, InDim>& x,
                                  const Tensor<T, Rows, OutDim>& d_out,
                                  std::size_t rows = Rows) {
    Tensor<T, Rows, InDim> dx;
    const std::size_t limit = rows < Rows ? rows : Rows;
    for (std::size_t r = 0; r < limit; ++r) {
      matvec_backward(x.data() + r * InDim, d_out.data() + r * OutDim,
                      dx.data() + r * InDim);
    }
    return dx;
  }

  void zero_grad() {
    dw_.fill(T{});
    db_.fill(T{});
  }

  void apply_sgd(T lr) {
    for (std::size_t i = 0; i < w_.size(); ++i) {
      w_[i] -= lr * dw_[i];
    }
    for (std::size_t i = 0; i < b_.size(); ++i) {
      b_[i] -= lr * db_[i];
    }
  }

  WeightTensor& weight() { return w_; }
  const WeightTensor& weight() const { return w_; }
  BiasTensor& bias() { return b_; }
  const BiasTensor& bias() const { return b_; }
  const WeightTensor& weight_grad() const { return dw_; }
  const BiasTensor& bias_grad() const { return db_; }

 private:
  WeightTensor w_;
  BiasTensor b_;
  WeightTensor dw_;
  BiasTensor db_;
};

// Normalizes each row over its `Dim` features, then scales and shifts.
template <typename T, std::size_t Dim>
class LayerNorm {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  static constexpr std::size_t dim = Dim;
  static constexpr std::size_t parameter_count = 2 * Dim;

  explicit LayerNorm(T eps = static_cast<T>(1e-5)) : eps_(eps) {
    gamma_.fill(T{1});
  }

  Tensor<T, Dim> forward(const Tensor<T, Dim>& x) const {
    Tensor<T, Dim> out;
    T mean{};
    for (std::size_t i = 0; i < Dim; ++i) {
      mean += x(i);
    }
    mean /= static_cast<T>(Dim);
    T var{};
    for (std::size_t i = 0; i < Dim; ++i) {
      const T c = x(i) - mean;
      var += c * c;
    }
    var /= static_cast<T>(Dim);
    const T rstd = T{1} / std::sqrt(var + eps_);
    for (std::size_t i = 0; i < Dim; ++i) {
      out(i) = gamma_(i) * (x(i) - mean) * rstd + beta_(i);
    }
    return out;
  }

  Tensor<T, Dim> backward(const Tensor<T, Dim>& x,
                          const Tensor<T, Dim>& d_out) {
    Tensor<T, Dim> dx;
    T mean{};
    for (std::size_t i = 0; i < Dim; ++i) {
      mean += x(i);
    }
    mean /= static_cast<T>(Dim);
    T var{};
    for (std::size_t i = 0; i < Dim; ++i) {
      const T c = x(i) - mean;
      var += c * c;
    }
    var /= static_cast<T>(Dim);
    const T rstd = T{1} / std::sqrt(var + eps_);

    // dy scaled by gamma, plus the two mean terms that normalization couples in
    T sum_dg{};
    T sum_dg_xhat{};
    for (std::size_t i = 0; i < Dim; ++i) {
      const T xhat = (x(i) - mean) * rstd;
      const T dg = d_out(i) * gamma_(i);
      dgamma_(i) += d_out(i) * xhat;
      dbeta_(i) += d_out(i);
      sum_dg += dg;
      sum_dg_xhat += dg * xhat;
    }
    const T inv_n = T{1} / static_cast<T>(Dim);
    for (std::size_t i = 0; i < Dim; ++i) {
      const T xhat = (x(i) - mean) * rstd;
      const T dg = d_out(i) * gamma_(i);
      dx(i) = rstd * (dg - inv_n * sum_dg - xhat * inv_n * sum_dg_xhat);
    }
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

  // One row, in place. Same arithmetic as the single-vector path above.
  ZF_ALWAYS_INLINE void normalize_row(const T* ZF_RESTRICT x,
                                      T* ZF_RESTRICT out) const {
    T mean{};
    for (std::size_t i = 0; i < Dim; ++i) {
      mean += x[i];
    }
    mean /= static_cast<T>(Dim);
    T var{};
    for (std::size_t i = 0; i < Dim; ++i) {
      const T c = x[i] - mean;
      var += c * c;
    }
    var /= static_cast<T>(Dim);
    const T rstd = T{1} / std::sqrt(var + eps_);
    const T* ZF_RESTRICT g = gamma_.data();
    const T* ZF_RESTRICT b = beta_.data();
    for (std::size_t i = 0; i < Dim; ++i) {
      out[i] = g[i] * (x[i] - mean) * rstd + b[i];
    }
  }

  ZF_ALWAYS_INLINE void normalize_row_backward(const T* ZF_RESTRICT x,
                                              const T* ZF_RESTRICT d_out,
                                              T* ZF_RESTRICT dx) {
    T mean{};
    for (std::size_t i = 0; i < Dim; ++i) {
      mean += x[i];
    }
    mean /= static_cast<T>(Dim);
    T var{};
    for (std::size_t i = 0; i < Dim; ++i) {
      const T c = x[i] - mean;
      var += c * c;
    }
    var /= static_cast<T>(Dim);
    const T rstd = T{1} / std::sqrt(var + eps_);

    const T* ZF_RESTRICT g = gamma_.data();
    T* ZF_RESTRICT dg = dgamma_.data();
    T* ZF_RESTRICT db = dbeta_.data();
    // dy scaled by gamma, plus the two mean terms normalization couples in
    T sum_dg{};
    T sum_dg_xhat{};
    for (std::size_t i = 0; i < Dim; ++i) {
      const T xhat = (x[i] - mean) * rstd;
      const T dgi = d_out[i] * g[i];
      dg[i] += d_out[i] * xhat;
      db[i] += d_out[i];
      sum_dg += dgi;
      sum_dg_xhat += dgi * xhat;
    }
    const T inv_n = T{1} / static_cast<T>(Dim);
    for (std::size_t i = 0; i < Dim; ++i) {
      const T xhat = (x[i] - mean) * rstd;
      const T dgi = d_out[i] * g[i];
      dx[i] = rstd * (dgi - inv_n * sum_dg - xhat * inv_n * sum_dg_xhat);
    }
  }

  void zero_grad() {
    dgamma_.fill(T{});
    dbeta_.fill(T{});
  }

  void apply_sgd(T lr) {
    for (std::size_t i = 0; i < Dim; ++i) {
      gamma_(i) -= lr * dgamma_(i);
      beta_(i) -= lr * dbeta_(i);
    }
  }

  Tensor<T, Dim>& gamma() { return gamma_; }
  const Tensor<T, Dim>& gamma() const { return gamma_; }
  Tensor<T, Dim>& beta() { return beta_; }
  const Tensor<T, Dim>& beta() const { return beta_; }
  const Tensor<T, Dim>& gamma_grad() const { return dgamma_; }
  const Tensor<T, Dim>& beta_grad() const { return dbeta_; }

 private:
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
  Tensor<T, Dim> dgamma_;
  Tensor<T, Dim> dbeta_;
  T eps_;
};

// Token id -> dense row. Gradients are scattered back per id.
template <typename T, std::size_t VocabSize, std::size_t EmbedDim>
class Embedding {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  using WeightTensor = Tensor<T, VocabSize, EmbedDim>;

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

  void zero_grad() { dw_.fill(T{}); }

  void apply_sgd(T lr) {
    for (std::size_t i = 0; i < w_.size(); ++i) {
      w_[i] -= lr * dw_[i];
    }
  }

  WeightTensor& weight() { return w_; }
  const WeightTensor& weight() const { return w_; }
  const WeightTensor& weight_grad() const { return dw_; }

 private:
  WeightTensor w_;
  WeightTensor dw_;
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

}  // namespace zf
