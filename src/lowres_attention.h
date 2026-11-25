#pragma once

// Scaled dot-product attention of one pass, on the CPU, for LowResCross and
// FocusCross: rows of HD = Heads x HeadDim, probabilities Heads x nq x nk.
//
// Two optional extras, both per key (nk):
//   key_bias  added to every score after the scale (FocusCross: log of a
//             key's relevance); attend_back sums the scores' gradient into
//             d_bias
//   dp_extra  a gradient on the probabilities themselves, the same for every
//             head and query row (FocusCross: what a key's relevance, read
//             off these probabilities, owes)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "layers.h"

namespace zf {

template <typename T, std::size_t Heads, std::size_t HeadDim, std::size_t MaxKeys>
struct LowResAttention {
  static constexpr std::size_t HD = Heads * HeadDim;

  // Overwrites ctx's nq rows.
  static void attend(const T* q, const T* k, const T* v, std::size_t nq, std::size_t nk, T* p, T* ctx,
                     const T* key_bias = nullptr) {
    const T scale = static_cast<T>(1.0 / std::sqrt(static_cast<double>(HeadDim)));
    if constexpr (HeadDim % 4 != 0) std::fill(ctx, ctx + nq * HD, T{});
    for (std::size_t h = 0; h < Heads; ++h) {
      const std::size_t o = h * HeadDim;
      for (std::size_t i = 0; i < nq; ++i) {
        T* pi = p + (h * nq + i) * nk;
        if constexpr (HeadDim % 4 == 0) {
          const Slice qi = Slice::load(q + i * HD + o);
          for (std::size_t j = 0; j < nk; ++j) pi[j] = qi.dot(k + j * HD + o) * scale;
        } else {
          const T* qi = q + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) pi[j] = dot(qi, k + j * HD + o) * scale;
        }
        if (key_bias != nullptr) {
          for (std::size_t j = 0; j < nk; ++j) pi[j] += key_bias[j];
        }
        T top = std::numeric_limits<T>::lowest();
        for (std::size_t j = 0; j < nk; ++j) top = std::max(top, pi[j]);
        T sum{};
        for (std::size_t j = 0; j < nk; ++j) sum += pi[j] = std::exp(pi[j] - top);
        const T inv = T{1} / sum;
        if constexpr (HeadDim % 4 == 0) {
          Slice ci = Slice::zero();
          for (std::size_t j = 0; j < nk; ++j) ci.add(pi[j] *= inv, v + j * HD + o);
          ci.store(ctx + i * HD + o);
        } else {
          T* ci = ctx + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) axpy(pi[j] *= inv, v + j * HD + o, ci);
        }
      }
    }
  }

  // Adds into d_q, d_k, d_v (and d_bias).
  static void attend_back(const T* q, const T* k, const T* v, const T* p, const T* d_ctx, std::size_t nq,
                          std::size_t nk, T* d_q, T* d_k, T* d_v, const T* dp_extra = nullptr,
                          T* d_bias = nullptr) {
    const T scale = static_cast<T>(1.0 / std::sqrt(static_cast<double>(HeadDim)));
    std::array<T, MaxKeys> dp{};
    for (std::size_t h = 0; h < Heads; ++h) {
      const std::size_t o = h * HeadDim;
      for (std::size_t i = 0; i < nq; ++i) {
        const T* pi = p + (h * nq + i) * nk;
        T pdp{};
        if constexpr (HeadDim % 4 == 0) {
          const Slice gi = Slice::load(d_ctx + i * HD + o);
          for (std::size_t j = 0; j < nk; ++j) {
            dp[j] = gi.dot(v + j * HD + o);
            if (dp_extra != nullptr) dp[j] += dp_extra[j];
            pdp += dp[j] * pi[j];
            gi.add_to(pi[j], d_v + j * HD + o);
          }
          const Slice qi = Slice::load(q + i * HD + o);
          Slice dqi = Slice::load(d_q + i * HD + o);
          for (std::size_t j = 0; j < nk; ++j) {
            const T dsr = pi[j] * (dp[j] - pdp);
            if (d_bias != nullptr) d_bias[j] += dsr;
            const T ds = dsr * scale;
            dqi.add(ds, k + j * HD + o);
            qi.add_to(ds, d_k + j * HD + o);
          }
          dqi.store(d_q + i * HD + o);
        } else {
          const T* gi = d_ctx + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) {
            dp[j] = dot(gi, v + j * HD + o);
            if (dp_extra != nullptr) dp[j] += dp_extra[j];
            pdp += dp[j] * pi[j];
            axpy(pi[j], gi, d_v + j * HD + o);
          }
          const T* qi = q + i * HD + o;
          T* dqi = d_q + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) {
            const T dsr = pi[j] * (dp[j] - pdp);
            if (d_bias != nullptr) d_bias[j] += dsr;
            const T ds = dsr * scale;
            axpy(ds, k + j * HD + o, dqi);
            axpy(ds, qi, d_k + j * HD + o);
          }
        }
      }
    }
  }

 private:
  // One head's slice of two rows: HeadDim long, four lanes at a time.
  static T dot(const T* a, const T* b) {
    T s{};
    for (std::size_t d = 0; d < HeadDim; ++d) s += a[d] * b[d];
    return s;
  }
  // y += w x over one head's slice.
  static void axpy(T w, const T* ZF_RESTRICT x, T* ZF_RESTRICT y) {
    for (std::size_t d = 0; d < HeadDim; ++d) y[d] += w * x[d];
  }

  // One head's slice of one row, held in registers while a loop runs over
  // the other rows (HeadDim % 4 == 0): attention's query, context and
  // gradient slices stay put instead of being reloaded for every key.
  struct Slice {
    using V = Vec4<T>;
    static constexpr std::size_t kLanes = HeadDim / 4;
    V v[kLanes];

    static Slice load(const T* p) {
      Slice s;
      for (std::size_t d = 0; d < kLanes; ++d) s.v[d] = load4(p + 4 * d);
      return s;
    }
    static Slice zero() {
      Slice s;
      for (std::size_t d = 0; d < kLanes; ++d) s.v[d] = V{};
      return s;
    }
    void store(T* p) const {
      for (std::size_t d = 0; d < kLanes; ++d) store4(p + 4 * d, v[d]);
    }
    T dot(const T* p) const {
      V acc = v[0] * load4(p);
      for (std::size_t d = 1; d < kLanes; ++d) acc += v[d] * load4(p + 4 * d);
      return hsum4(acc);
    }
    // *this += w p
    void add(T w, const T* p) {
      const V s{w, w, w, w};
      for (std::size_t d = 0; d < kLanes; ++d) v[d] += s * load4(p + 4 * d);
    }
    // p += w *this
    void add_to(T w, T* p) const {
      const V s{w, w, w, w};
      for (std::size_t d = 0; d < kLanes; ++d) store4(p + 4 * d, load4(p + 4 * d) + s * v[d]);
    }
  };
};

}  // namespace zf
