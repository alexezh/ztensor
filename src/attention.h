#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "add.h"
#include "layers.h"
#include "tensor.h"

namespace zf {

// Multi-head self attention over a fixed-length sequence.
//
// Sequence length and model width are template parameters, so every projection
// is shape-checked at compile time. The head count only partitions the model
// width, which does not change any parameter shape, so it is a runtime
// constructor argument (ModelDim % heads must be 0). Causal masking and the
// valid length are runtime settings for the same reason.
//
// `valid_length` marks how much of the sequence is real, which is what lets a
// padded batch of shorter sequences share one compile-time SeqLen. Keys at or
// past it are never attended to, so a padded position cannot influence a real
// one, and padded rows are not computed at all: their outputs stay zero, their
// attention weights stay empty, and they contribute no gradient. They
// contributed none before either -- a padded row's incoming gradient is zero
// -- so skipping them changes no result, only the work done.
template <typename T, std::size_t SeqLen, std::size_t ModelDim>
class MultiHeadSelfAttention {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");

 public:
  using value_type = T;
  using Sequence = Tensor<T, SeqLen, ModelDim>;
  using Scores = Tensor<T, SeqLen, SeqLen>;

  static constexpr std::size_t seq_len = SeqLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t parameter_count =
      4 * Linear<T, ModelDim, ModelDim>::parameter_count;

  explicit MultiHeadSelfAttention(std::size_t heads, bool causal = false)
      : heads_(heads), causal_(causal) {
    if (heads == 0 || ModelDim % heads != 0) {
      throw std::invalid_argument(
          "head count " + std::to_string(heads) +
          " must be non-zero and divide model dim " + std::to_string(ModelDim));
    }
    attn_.assign(heads_, Scores{});
  }

  std::size_t heads() const { return heads_; }
  std::size_t head_dim() const { return ModelDim / heads_; }
  bool causal() const { return causal_; }
  void set_causal(bool causal) { causal_ = causal; }

  std::size_t valid_length() const { return valid_length_; }

  // Throws std::out_of_range unless 1 <= length <= SeqLen. A length of zero
  // would leave a row with nothing to attend to and no softmax to take.
  void set_valid_length(std::size_t length) {
    if (length == 0 || length > SeqLen) {
      throw std::out_of_range("valid length " + std::to_string(length) +
                              " outside [1, " + std::to_string(SeqLen) + "]");
    }
    valid_length_ = length;
  }

  template <typename Rng>
  void randomize(Rng& rng) {
    wq_.randomize(rng);
    wk_.randomize(rng);
    wv_.randomize(rng);
    wo_.randomize(rng);
  }

  // The kernels below only ever touch the first `valid_length` rows. Padded
  // rows are read by nobody -- a padded key is masked out of every softmax,
  // and a padded query's output is never looked at -- and they receive no
  // gradient, so computing them was waste proportional to how much of the
  // compile-time sequence a given input actually uses.
  Sequence forward(const Sequence& x) {
    const std::size_t hd = head_dim();
    const std::size_t rows = valid_length_;
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    x_ = x;
    // The projections need only the rows that exist.
    q_ = wq_.template forward<SeqLen>(x, rows);
    k_ = wk_.template forward<SeqLen>(x, rows);
    v_ = wv_.template forward<SeqLen>(x, rows);
    ctx_.fill(T{});

    const T* ZF_RESTRICT qp = q_.data();
    const T* ZF_RESTRICT kp = k_.data();
    const T* ZF_RESTRICT vp = v_.data();
    T* ZF_RESTRICT cp = ctx_.data();

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      Scores& a = attn_[h];
      T* ZF_RESTRICT ap = a.data();
      for (std::size_t i = 0; i < rows; ++i) {
        const std::size_t limit = key_limit(i);
        const T* ZF_RESTRICT qi = qp + i * ModelDim + off;
        T* ZF_RESTRICT arow = ap + i * SeqLen;

        // Scores, four keys at a time against one query slice, for the same
        // reason Linear does four outputs at once: the query stays in
        // registers and four independent accumulators keep the pipeline fed.
        std::size_t j = 0;
        for (; j + 4 <= limit; j += 4) {
          const T* ZF_RESTRICT k0 = kp + (j + 0) * ModelDim + off;
          const T* ZF_RESTRICT k1 = kp + (j + 1) * ModelDim + off;
          const T* ZF_RESTRICT k2 = kp + (j + 2) * ModelDim + off;
          const T* ZF_RESTRICT k3 = kp + (j + 3) * ModelDim + off;
          T s0{};
          T s1{};
          T s2{};
          T s3{};
          for (std::size_t d = 0; d < hd; ++d) {
            const T qv = qi[d];
            s0 += k0[d] * qv;
            s1 += k1[d] * qv;
            s2 += k2[d] * qv;
            s3 += k3[d] * qv;
          }
          arow[j + 0] = s0 * inv_scale;
          arow[j + 1] = s1 * inv_scale;
          arow[j + 2] = s2 * inv_scale;
          arow[j + 3] = s3 * inv_scale;
        }
        for (; j < limit; ++j) {
          const T* ZF_RESTRICT kj = kp + j * ModelDim + off;
          T sum{};
          for (std::size_t d = 0; d < hd; ++d) {
            sum += kj[d] * qi[d];
          }
          arow[j] = sum * inv_scale;
        }

        softmax_inplace(arow, limit);
        for (std::size_t m = limit; m < SeqLen; ++m) {
          arow[m] = T{};
        }

        // Context, accumulated over keys rather than one output dimension at
        // a time: the inner loop then walks a row of V contiguously and has
        // no reduction in it at all.
        T* ZF_RESTRICT ci = cp + i * ModelDim + off;
        for (std::size_t m = 0; m < limit; ++m) {
          const T weight = arow[m];
          const T* ZF_RESTRICT vm = vp + m * ModelDim + off;
          for (std::size_t d = 0; d < hd; ++d) {
            ci[d] += weight * vm[d];
          }
        }
      }
    }
    return wo_.template forward<SeqLen>(ctx_, rows);
  }

  // Returns dL/dx. Must follow a matching forward().
  Sequence backward(const Sequence& d_out) {
    const std::size_t hd = head_dim();
    const std::size_t rows = valid_length_;
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    const Sequence d_ctx = wo_.template backward<SeqLen>(ctx_, d_out, rows);

    Sequence dq;
    Sequence dk;
    Sequence dv;
    const T* ZF_RESTRICT qp = q_.data();
    const T* ZF_RESTRICT kp = k_.data();
    const T* ZF_RESTRICT vp = v_.data();
    const T* ZF_RESTRICT gp = d_ctx.data();
    T* ZF_RESTRICT dqp = dq.data();
    T* ZF_RESTRICT dkp = dk.data();
    T* ZF_RESTRICT dvp = dv.data();

    // Scratch for one row of attention-weight gradient, hoisted out of both
    // loops: inside them it was a fresh heap allocation per head per row.
    Tensor<T, SeqLen> d_attn;
    T* ZF_RESTRICT dap = d_attn.data();

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      const Scores& a = attn_[h];
      const T* ZF_RESTRICT ap = a.data();
      for (std::size_t i = 0; i < rows; ++i) {
        const std::size_t limit = key_limit(i);
        const T* ZF_RESTRICT arow = ap + i * SeqLen;
        const T* ZF_RESTRICT gi = gp + i * ModelDim + off;

        // dL/d(attention weights), and the value gradient they feed.
        for (std::size_t j = 0; j < limit; ++j) {
          const T* ZF_RESTRICT vj = vp + j * ModelDim + off;
          T* ZF_RESTRICT dvj = dvp + j * ModelDim + off;
          const T weight = arow[j];
          T acc{};
          for (std::size_t d = 0; d < hd; ++d) {
            acc += gi[d] * vj[d];
            dvj[d] += weight * gi[d];
          }
          dap[j] = acc;
        }

        // softmax jacobian: ds_j = p_j * (dp_j - sum_m p_m dp_m)
        T dot_pd{};
        for (std::size_t j = 0; j < limit; ++j) {
          dot_pd += arow[j] * dap[j];
        }
        T* ZF_RESTRICT dqi = dqp + i * ModelDim + off;
        for (std::size_t j = 0; j < limit; ++j) {
          const T ds = arow[j] * (dap[j] - dot_pd) * inv_scale;
          const T* ZF_RESTRICT kj = kp + j * ModelDim + off;
          T* ZF_RESTRICT dkj = dkp + j * ModelDim + off;
          const T* ZF_RESTRICT qi = qp + i * ModelDim + off;
          for (std::size_t d = 0; d < hd; ++d) {
            dqi[d] += ds * kj[d];
            dkj[d] += ds * qi[d];
          }
        }
      }
    }

    Sequence dx = wq_.template backward<SeqLen>(x_, dq, rows);
    add_into(dx, wk_.template backward<SeqLen>(x_, dk, rows));
    add_into(dx, wv_.template backward<SeqLen>(x_, dv, rows));
    return dx;
  }

  void zero_grad() {
    wq_.zero_grad();
    wk_.zero_grad();
    wv_.zero_grad();
    wo_.zero_grad();
  }

  void apply_sgd(T lr) {
    wq_.apply_sgd(lr);
    wk_.apply_sgd(lr);
    wv_.apply_sgd(lr);
    wo_.apply_sgd(lr);
  }

  Linear<T, ModelDim, ModelDim>& query_proj() { return wq_; }
  Linear<T, ModelDim, ModelDim>& key_proj() { return wk_; }
  Linear<T, ModelDim, ModelDim>& value_proj() { return wv_; }
  Linear<T, ModelDim, ModelDim>& output_proj() { return wo_; }

  // Attention weights from the last forward(), per head. Only the first
  // valid_length rows are populated; the rest are left zero because no such
  // query exists.
  const Scores& attention_weights(std::size_t head) const {
    return attn_.at(head);
  }

 private:
  // Keys query `i` may attend to: never past the valid length, and never
  // ahead of itself when causal. At least one, so softmax always has a row.
  std::size_t key_limit(std::size_t i) const {
    const std::size_t causal_limit = causal_ ? i + 1 : SeqLen;
    const std::size_t limit =
        causal_limit < valid_length_ ? causal_limit : valid_length_;
    return limit == 0 ? 1 : limit;
  }

  std::size_t heads_;
  bool causal_;
  std::size_t valid_length_ = SeqLen;

  Linear<T, ModelDim, ModelDim> wq_;
  Linear<T, ModelDim, ModelDim> wk_;
  Linear<T, ModelDim, ModelDim> wv_;
  Linear<T, ModelDim, ModelDim> wo_;

  // activations cached by forward() for use by backward()
  Sequence x_;
  Sequence q_;
  Sequence k_;
  Sequence v_;
  Sequence ctx_;
  std::vector<Scores> attn_;
};

}  // namespace zf
