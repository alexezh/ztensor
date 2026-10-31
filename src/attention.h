#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "add.h"
#include "layers.h"
#include "tensor.h"

namespace zf {

// --- attention kernels, shared by self and cross attention ---
//
// Pointers are to one head's slice of a row; consecutive rows are ModelDim
// apart. Keys and values are the `limit` rows starting at k / v.

// arow[j] = (q . k_j) * scale for j < limit. Four keys at a time against one
// query slice, for the same reason Linear does four outputs at once: the query
// stays in registers and four independent accumulators keep the pipeline fed.
template <std::size_t ModelDim, typename T>
ZF_ALWAYS_INLINE void score_row(const T* ZF_RESTRICT q, const T* ZF_RESTRICT k,
                                std::size_t limit, std::size_t hd, T scale,
                                T* ZF_RESTRICT arow) {
  std::size_t j = 0;
  for (; j + 4 <= limit; j += 4) {
    const T* ZF_RESTRICT k0 = k + (j + 0) * ModelDim;
    const T* ZF_RESTRICT k1 = k + (j + 1) * ModelDim;
    const T* ZF_RESTRICT k2 = k + (j + 2) * ModelDim;
    const T* ZF_RESTRICT k3 = k + (j + 3) * ModelDim;
    T s0{};
    T s1{};
    T s2{};
    T s3{};
    for (std::size_t d = 0; d < hd; ++d) {
      const T qv = q[d];
      s0 += k0[d] * qv;
      s1 += k1[d] * qv;
      s2 += k2[d] * qv;
      s3 += k3[d] * qv;
    }
    arow[j + 0] = s0 * scale;
    arow[j + 1] = s1 * scale;
    arow[j + 2] = s2 * scale;
    arow[j + 3] = s3 * scale;
  }
  for (; j < limit; ++j) {
    const T* ZF_RESTRICT kj = k + j * ModelDim;
    T sum{};
    for (std::size_t d = 0; d < hd; ++d) sum += kj[d] * q[d];
    arow[j] = sum * scale;
  }
}

// Backward of one query row of one head: given its weights `arow` and the
// context gradient `g`, adds into dq (this row), dk and dv (the key rows).
// `dap` is scratch of at least `limit`.
template <std::size_t ModelDim, typename T>
ZF_ALWAYS_INLINE void attend_backward(
    const T* ZF_RESTRICT arow, const T* ZF_RESTRICT g, const T* ZF_RESTRICT q,
    const T* ZF_RESTRICT k, const T* ZF_RESTRICT v, std::size_t limit,
    std::size_t hd, T scale, T* ZF_RESTRICT dap, T* ZF_RESTRICT dq,
    T* ZF_RESTRICT dk, T* ZF_RESTRICT dv) {
  // dL/d(attention weights), and the value gradient they feed.
  for (std::size_t j = 0; j < limit; ++j) {
    const T* ZF_RESTRICT vj = v + j * ModelDim;
    T* ZF_RESTRICT dvj = dv + j * ModelDim;
    const T weight = arow[j];
    T acc{};
    for (std::size_t d = 0; d < hd; ++d) {
      acc += g[d] * vj[d];
      dvj[d] += weight * g[d];
    }
    dap[j] = acc;
  }
  // softmax jacobian: ds_j = p_j * (dp_j - sum_m p_m dp_m)
  T dot_pd{};
  for (std::size_t j = 0; j < limit; ++j) dot_pd += arow[j] * dap[j];
  for (std::size_t j = 0; j < limit; ++j) {
    const T ds = arow[j] * (dap[j] - dot_pd) * scale;
    const T* ZF_RESTRICT kj = k + j * ModelDim;
    T* ZF_RESTRICT dkj = dk + j * ModelDim;
    for (std::size_t d = 0; d < hd; ++d) {
      dq[d] += ds * kj[d];
      dkj[d] += ds * q[d];
    }
  }
}

// Multi-head self attention over a fixed-length sequence, or over a batch of
// them.
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
//
// Batch: `Batch` independent sequences stacked row-wise, Batch * SeqLen rows,
// example b in rows [b * SeqLen, (b + 1) * SeqLen) with its own valid length.
// Attention is block-diagonal -- a row only ever sees keys of its own example
// -- so a batch computes exactly what Batch separate calls would, while the
// projections run over all real rows at once. Parameter gradients accumulate
// over the whole batch; one apply_sgd then takes one step for all of it.
template <typename T, std::size_t SeqLen, std::size_t ModelDim,
          std::size_t Batch = 1>
class MultiHeadSelfAttention {
  static_assert(std::is_floating_point_v<T>,
                "trainable layers require a floating point element type");
  static_assert(Batch > 0, "a batch holds at least one example");

 public:
  using value_type = T;
  static constexpr std::size_t kRows = Batch * SeqLen;
  using Sequence = Tensor<T, kRows, ModelDim>;
  // Row = query row (all examples stacked), column = key within its example.
  using Scores = Tensor<T, kRows, SeqLen>;

  static constexpr std::size_t seq_len = SeqLen;
  static constexpr std::size_t model_dim = ModelDim;
  static constexpr std::size_t batch = Batch;
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
    set_valid_length(SeqLen);
  }

  std::size_t heads() const { return heads_; }
  std::size_t head_dim() const { return ModelDim / heads_; }
  bool causal() const { return causal_; }
  void set_causal(bool causal) { causal_ = causal; }

  // Example 0's valid length; with Batch == 1, the valid length.
  std::size_t valid_length() const { return spans_[0].count; }
  std::size_t valid_length(std::size_t b) const { return spans_.at(b).count; }

  // Throws std::out_of_range unless 1 <= length <= SeqLen. A length of zero
  // would leave a row with nothing to attend to and no softmax to take.
  // The one-argument form sets every example.
  void set_valid_length(std::size_t length) {
    for (std::size_t b = 0; b < Batch; ++b) set_valid_length(b, length);
  }
  void set_valid_length(std::size_t b, std::size_t length) {
    if (length == 0 || length > SeqLen) {
      throw std::out_of_range("valid length " + std::to_string(length) +
                              " outside [1, " + std::to_string(SeqLen) + "]");
    }
    spans_.at(b) = RowSpan{b * SeqLen, length};
  }

  // The real rows of every example, for the row-wise layers around this one.
  RowSpans row_spans() const { return spans_; }

  template <typename Rng>
  void randomize(Rng& rng) {
    wq_.randomize(rng);
    wk_.randomize(rng);
    wv_.randomize(rng);
    wo_.randomize(rng);
  }

  // The kernels below only ever touch real rows. Padded rows are read by
  // nobody -- a padded key is masked out of every softmax, and a padded
  // query's output is never looked at -- and they receive no gradient.
  Sequence forward(const Sequence& x) {
    const std::size_t hd = head_dim();
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    x_ = x;
    q_.fill(T{});
    k_.fill(T{});
    v_.fill(T{});
    wq_.forward_spans(x, q_, spans_);
    wk_.forward_spans(x, k_, spans_);
    wv_.forward_spans(x, v_, spans_);
    ctx_.fill(T{});

    const T* ZF_RESTRICT qp = q_.data();
    const T* ZF_RESTRICT kp = k_.data();
    const T* ZF_RESTRICT vp = v_.data();
    T* ZF_RESTRICT cp = ctx_.data();

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      T* ZF_RESTRICT ap = attn_[h].data();
      for (std::size_t b = 0; b < Batch; ++b) {
        const std::size_t base = b * SeqLen;
        const std::size_t valid = spans_[b].count;
        const T* ZF_RESTRICT kb = kp + base * ModelDim + off;
        const T* ZF_RESTRICT vb = vp + base * ModelDim + off;
        for (std::size_t i = 0; i < valid; ++i) {
          const std::size_t limit = key_limit(i, valid);
          const T* ZF_RESTRICT qi = qp + (base + i) * ModelDim + off;
          T* ZF_RESTRICT arow = ap + (base + i) * SeqLen;
          score_row<ModelDim>(qi, kb, limit, hd, inv_scale, arow);
          softmax_inplace(arow, limit);
          for (std::size_t m = limit; m < SeqLen; ++m) arow[m] = T{};
          // Context, accumulated over keys rather than one output dimension
          // at a time: the inner loop then walks a row of V contiguously and
          // has no reduction in it at all.
          T* ZF_RESTRICT ci = cp + (base + i) * ModelDim + off;
          for (std::size_t m = 0; m < limit; ++m) {
            const T weight = arow[m];
            const T* ZF_RESTRICT vm = vb + m * ModelDim;
            for (std::size_t d = 0; d < hd; ++d) ci[d] += weight * vm[d];
          }
        }
        // A padded query has no weights.
        for (std::size_t i = valid; i < SeqLen; ++i) {
          T* ZF_RESTRICT arow = ap + (base + i) * SeqLen;
          for (std::size_t m = 0; m < SeqLen; ++m) arow[m] = T{};
        }
      }
    }
    Sequence out;
    wo_.forward_spans(ctx_, out, spans_);
    return out;
  }

  // Returns dL/dx. Must follow a matching forward().
  Sequence backward(const Sequence& d_out) {
    const std::size_t hd = head_dim();
    const T inv_scale = T{1} / std::sqrt(static_cast<T>(hd));

    Sequence d_ctx;
    wo_.backward_spans(ctx_, d_out, d_ctx, spans_);

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

    // Scratch for one row of attention-weight gradient.
    Tensor<T, SeqLen> d_attn;
    T* ZF_RESTRICT dap = d_attn.data();

    for (std::size_t h = 0; h < heads_; ++h) {
      const std::size_t off = h * hd;
      const T* ZF_RESTRICT ap = attn_[h].data();
      for (std::size_t b = 0; b < Batch; ++b) {
        const std::size_t base = b * SeqLen;
        const std::size_t valid = spans_[b].count;
        for (std::size_t i = 0; i < valid; ++i) {
          const std::size_t row = base + i;
          const std::size_t limit = key_limit(i, valid);
          attend_backward<ModelDim>(ap + row * SeqLen, gp + row * ModelDim + off,
                          qp + row * ModelDim + off, kp + base * ModelDim + off,
                          vp + base * ModelDim + off, limit, hd, inv_scale, dap,
                          dqp + row * ModelDim + off, dkp + base * ModelDim + off,
                          dvp + base * ModelDim + off);
        }
      }
    }

    Sequence dx;
    wq_.backward_spans(x_, dq, dx, spans_);
    wk_.backward_spans(x_, dk, dx, spans_);
    wv_.backward_spans(x_, dv, dx, spans_);
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

  // Attention weights from the last forward(), per head. Rows of padded
  // queries are zero because no such query exists.
  const Scores& attention_weights(std::size_t head) const {
    return attn_.at(head);
  }

 private:
  // Keys query `i` may attend to: never past the valid length, and never
  // ahead of itself when causal. At least one, so softmax always has a row.
  std::size_t key_limit(std::size_t i, std::size_t valid) const {
    const std::size_t causal_limit = causal_ ? i + 1 : SeqLen;
    const std::size_t limit = causal_limit < valid ? causal_limit : valid;
    return limit == 0 ? 1 : limit;
  }

  std::size_t heads_;
  bool causal_;
  std::array<RowSpan, Batch> spans_{};

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
