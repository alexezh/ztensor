#pragma once

// FocusCross on the CPU (the facade is focuscross.h): the activations of a
// batch over the facade's weights.
//
// As in LowResCrossCpu, a batch's query rows are stacked into one matrix and
// so are its workspace rows, every projection is one matrix product over
// the batch, and attention runs per pass over that pass's rows only. The
// workspace rows are the pass's own and change at every level, so there is
// no cache of their K and V.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "focuscross_weights.h"
#include "layers.h"
#include "lowres_attention.h"
#include "lowrescross_layout.h"
#include "tensor.h"
#include "thread_pool.h"

namespace zf {

template <typename T, typename Layout, std::size_t Q, std::size_t M, std::size_t Heads,
          std::size_t HeadDim, std::size_t Ffn>
class FocusCrossCpu {
 public:
  static constexpr std::size_t HD = Heads * HeadDim;
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  static constexpr std::size_t kHiddenOffset = Layout::kStateOffset;
  using Input = typename Layout::Input;
  using StateGrad = typename Layout::StateGrad;
  using Pass = LowResPass<Input>;
  using Weights = FocusCrossWeights<T, Layout, HD, Ffn>;
  using WLayer = typename Weights::Layer;
  using WFocus = typename Weights::Focus;

  void set_pool(ThreadPool* pool) { pool_ = pool; }

  void forward_batch(Weights& w, const std::vector<Pass>& batch) {
    nb_ = batch.size();
    qo_.assign(nb_ + 1, 0);
    mo_.assign(nb_ + 1, 0);
    so_.assign(nb_ + 1, 0);
    co_.assign(nb_ + 1, 0);
    wo_.assign(nb_ + 1, 0);
    queries_.clear();
    memory_.clear();
    for (std::size_t b = 0; b < nb_; ++b) {
      const std::size_t nq = batch[b].queries->size(), nm = batch[b].memory->size();
      if (nq > Q || nm > M) throw std::length_error("FocusCross overflow");
      qo_[b + 1] = qo_[b] + nq;
      mo_[b + 1] = mo_[b] + nm;
      so_[b + 1] = so_[b] + Heads * nq * nq;
      co_[b + 1] = co_[b] + Heads * nq * nm;
      wo_[b + 1] = wo_[b] + Heads * nm * nm;
      queries_.insert(queries_.end(), batch[b].queries->begin(), batch[b].queries->end());
      memory_.insert(memory_.end(), batch[b].memory->begin(), batch[b].memory->end());
    }
    rows_ = qo_[nb_];
    mrows_ = mo_[nb_];
    // The rows, their read masks, and where the layers may write: the query
    // rows' cross attention only in passes that have a workspace.
    zeroed(&x0_, rows_ * kStream);
    zeroed(&qmask_, rows_ * kStream);
    zeroed(&mask_, rows_ * kHidden);
    zeroed(&cmask_, rows_ * kHidden);
    for (std::size_t b = 0; b < nb_; ++b) {
      for (std::size_t i = qo_[b]; i < qo_[b + 1]; ++i) {
        w.put(x0_.data() + i * kStream, queries_[i]);
        Layout::put_mask(qmask_.data() + i * kStream, queries_[i]);
        Layout::put_write_mask(mask_.data() + i * kHidden, queries_[i]);
        if (mo_[b + 1] > mo_[b]) Layout::put_write_mask(cmask_.data() + i * kHidden, queries_[i]);
      }
    }
    zeroed(&m0_, mrows_ * kStream);
    zeroed(&mmask_, mrows_ * kStream);
    zeroed(&mwmask_, mrows_ * kHidden);
    for (std::size_t j = 0; j < mrows_; ++j) {
      w.put(m0_.data() + j * kStream, memory_[j]);
      Layout::put_mask(mmask_.data() + j * kStream, memory_[j]);
      Layout::put_write_mask(mwmask_.data() + j * kHidden, memory_[j]);
    }
    // Before the first cross attention every workspace row counts.
    ones_.assign(mrows_, T{1});
    acts_.resize(w.layers.size());
    const std::vector<T>* x = &x0_;
    const std::vector<T>* m = &m0_;
    const std::vector<T>* r = &ones_;
    for (std::size_t li = 0; li < acts_.size(); ++li) {
      level_forward(w.layers[li], w.focus[li], acts_[li], *x, *m, *r);
      x = &acts_[li].x3;
      m = &acts_[li].w1;
      r = &acts_[li].r;
    }
    final_ = x;
    mfinal_ = m;
    zeroed(&d_m_extra_, mrows_ * kStream);
  }

  // --- the batch just run ---

  std::size_t batch_size() const { return nb_; }
  std::size_t query_size(std::size_t b) const { return qo_[b + 1] - qo_[b]; }
  const T* hidden_row(std::size_t b, std::size_t i) const {
    return final_->data() + (qo_[b] + i) * kStream + kHiddenOffset;
  }
  // Pass b's workspace rows after the last level.
  std::size_t memory_size(std::size_t b) const { return mo_[b + 1] - mo_[b]; }
  const T* memory_rows(std::size_t b) const { return mfinal_->data() + mo_[b] * kStream; }
  T* memory_grad(std::size_t b) { return d_m_extra_.data() + mo_[b] * kStream; }
  // Pass b's workspace relevance after level `level`'s cross attention.
  const T* relevance(std::size_t b, std::size_t level) const { return acts_.at(level).r.data() + mo_[b]; }

  std::vector<std::vector<StateGrad>> backward_batch(Weights& w, const std::vector<Tensor<T, Q, kHidden>>& d_readout) {
    if (d_readout.size() != nb_) throw std::invalid_argument("FocusCross: one gradient per pass");
    zeroed(&d_, rows_ * kStream);
    for (std::size_t b = 0; b < nb_; ++b) {
      for (std::size_t i = 0; i < query_size(b); ++i) {
        T* d = d_.data() + (qo_[b] + i) * kStream + kHiddenOffset;
        for (std::size_t k = 0; k < kHidden; ++k) d[k] = d_readout[b](i, k);
      }
    }
    d_m_ = d_m_extra_;
    zeroed(&dr_, mrows_);  // the last level's relevance reaches nothing
    for (std::size_t li = acts_.size(); li-- > 0;) {
      level_backward(w.layers[li], w.focus[li], acts_[li], li == 0 ? x0_ : acts_[li - 1].x3,
                     li == 0 ? m0_ : acts_[li - 1].w1);
    }
    for (std::size_t i = 0; i < rows_; ++i) w.identity_backward(queries_[i], d_.data() + i * kStream);
    for (std::size_t j = 0; j < mrows_; ++j) w.identity_backward(memory_[j], d_m_.data() + j * kStream);
    zeroed(&d_m_extra_, mrows_ * kStream);
    std::vector<std::vector<StateGrad>> out(nb_);
    for (std::size_t b = 0; b < nb_; ++b) {
      out[b].reserve(query_size(b));
      for (std::size_t i = 0; i < query_size(b); ++i) {
        const std::size_t row = qo_[b] + i;
        out[b].push_back(Layout::state_grad(d_.data() + row * kStream + kHiddenOffset, queries_[row]));
      }
    }
    return out;
  }

 private:
  using RowStats = typename LayerNorm<T, kStream>::RowStats;
  using Attn = LowResAttention<T, Heads, HeadDim, (Q > M ? Q : M)>;

  // A level's activations, row-stacked: rows_ query rows, mrows_ workspace rows.
  struct Acts {
    // the workspace's self attention, gated by the relevance it came in with
    std::vector<T> rin;                       // mrows
    std::vector<T> wn, w1;                    // mrows x kStream
    std::vector<T> qw, kw, vw, ctxw;          // mrows x HD
    std::vector<T> uw;                        // mrows x kHidden
    std::vector<T> pw;                        // per pass Heads x nm x nm
    // the queries' self attention, the cross attention, the feed-forward
    std::vector<T> xn1, x1, xn2, x2, xn3, x3;  // rows x kStream
    std::vector<T> mn;                         // mrows x kStream
    std::vector<T> q, k, v, ctx, qc, ctx_c;    // rows x HD
    std::vector<T> kc, vc;                     // mrows x HD
    std::vector<T> ps, pc;
    std::vector<T> pre, act;                   // rows x Ffn
    std::vector<T> r;                          // mrows: the relevance it leaves
    std::vector<RowStats> sw, s1, s2, sm, s3;
  };

  static void zeroed(std::vector<T>* v, std::size_t n) { v->assign(n, T{}); }

  template <typename L>
  static void linear(const L& l, const std::vector<T>& x, std::size_t rows, std::vector<T>* out) {
    out->resize(rows * L::out_dim);
    if (rows > 0) l.forward_into(x.data(), out->data(), rows);
  }
  // Overwrites *dx with dL/dx; adds into the layer's gradients.
  template <typename L>
  static void linear_back(L& l, const std::vector<T>& x, const std::vector<T>& d_out, std::size_t rows,
                          std::vector<T>* dx) {
    zeroed(dx, rows * L::in_dim);
    if (rows > 0) l.backward_into(x.data(), d_out.data(), dx->data(), rows);
  }
  // Adds dL/dx into *dx.
  template <typename L>
  static void linear_back_add(L& l, const std::vector<T>& x, const std::vector<T>& d_out, std::size_t rows,
                              std::vector<T>* dx) {
    if (rows > 0) l.backward_into(x.data(), d_out.data(), dx->data(), rows);
  }
  void norm(const LayerNorm<T, kStream>& n, const std::vector<T>& x, std::size_t rows, const T* mask,
            std::vector<RowStats>* stats, std::vector<T>* out) const {
    stats->resize(rows);
    out->resize(rows * kStream);
    if (rows == 0) return;
    n.stats_rows(x.data(), mask, rows, stats->data(), pool_);
    n.forward_rows_from(stats->data(), x.data(), out->data(), rows, mask, pool_);
  }
  // Adds dL/dx into *dx.
  void norm_back(LayerNorm<T, kStream>& n, const std::vector<T>& x, const std::vector<T>& d_out,
                 std::size_t rows, const T* mask, const std::vector<RowStats>& stats, std::vector<T>* dx) {
    if (rows == 0) return;
    row_.resize(rows * kStream);
    n.backward_rows_from(stats.data(), x.data(), d_out.data(), row_.data(), rows, mask, pool_);
    for (std::size_t i = 0; i < rows * kStream; ++i) (*dx)[i] += row_[i];
  }
  template <typename F>
  void each_pass(F&& f) const {
    parallel_for(pool_, nb_, 1, [&](std::size_t, std::size_t b0, std::size_t b1) {
      for (std::size_t b = b0; b < b1; ++b) f(b);
    });
  }

  // x (n x kStream) += gate · mask · u (n x kHidden); no gate: 1.
  static void add_masked(std::vector<T>* x, const std::vector<T>& u, const std::vector<T>& mask, std::size_t n,
                         const T* gate = nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
      T* h = x->data() + i * kStream + kHiddenOffset;
      const T* ui = u.data() + i * kHidden;
      const T* mi = mask.data() + i * kHidden;
      const T g = gate != nullptr ? gate[i] : T{1};
      for (std::size_t k = 0; k < kHidden; ++k) h[k] += g * mi[k] * ui[k];
    }
  }
  static void masked_hidden_grad(const std::vector<T>& d, const std::vector<T>& mask, std::size_t n,
                                 std::vector<T>* g) {
    g->resize(n * kHidden);
    for (std::size_t i = 0; i < n; ++i) {
      const T* di = d.data() + i * kStream + kHiddenOffset;
      const T* mi = mask.data() + i * kHidden;
      T* gi = g->data() + i * kHidden;
      for (std::size_t k = 0; k < kHidden; ++k) gi[k] = mi[k] * di[k];
    }
  }

  // x, m: the level's query and workspace rows; r: the relevance from the
  // level before (ones at the first).
  void level_forward(const WLayer& wl, const WFocus& wf, Acts& l, const std::vector<T>& x, const std::vector<T>& m,
                     const std::vector<T>& r) {
    // workspace self attention: w1 = m + r · mask · O_w(attn(..., keys biased by log r))
    l.rin = r;
    norm(wf.nw, m, mrows_, mmask_.data(), &l.sw, &l.wn);
    linear(wf.qw, l.wn, mrows_, &l.qw);
    linear(wf.kw, l.wn, mrows_, &l.kw);
    linear(wf.vw, l.wn, mrows_, &l.vw);
    bias_.resize(mrows_);
    for (std::size_t j = 0; j < mrows_; ++j) bias_[j] = static_cast<T>(std::log(r[j] + kFocusEps));
    l.ctxw.resize(mrows_ * HD);
    l.pw.resize(wo_[nb_]);
    each_pass([&](std::size_t b) {
      const std::size_t o = mo_[b] * HD, n = memory_size(b);
      if (n == 0) return;
      Attn::attend(l.qw.data() + o, l.kw.data() + o, l.vw.data() + o, n, n, l.pw.data() + wo_[b],
                   l.ctxw.data() + o, bias_.data() + mo_[b]);
    });
    linear(wf.ow, l.ctxw, mrows_, &l.uw);
    l.w1 = m;
    add_masked(&l.w1, l.uw, mwmask_, mrows_, r.data());
    // query self attention
    norm(wl.n1, x, rows_, qmask_.data(), &l.s1, &l.xn1);
    linear(wl.qs, l.xn1, rows_, &l.q);
    linear(wl.ks, l.xn1, rows_, &l.k);
    linear(wl.vs, l.xn1, rows_, &l.v);
    l.ctx.resize(rows_ * HD);
    l.ps.resize(so_[nb_]);
    each_pass([&](std::size_t b) {
      const std::size_t o = qo_[b] * HD, n = query_size(b);
      Attn::attend(l.q.data() + o, l.k.data() + o, l.v.data() + o, n, n, l.ps.data() + so_[b], l.ctx.data() + o);
    });
    linear(wl.os, l.ctx, rows_, &u_);
    l.x1 = x;
    add_masked(&l.x1, u_, mask_, rows_);
    // cross attention into the workspace, and the relevance it reads:
    // r_j = 1 - exp(-s_j), s_j the attention row j got, summed over the
    // pass's query rows, averaged over heads
    norm(wl.n2, l.x1, rows_, qmask_.data(), &l.s2, &l.xn2);
    norm(wl.nm, l.w1, mrows_, mmask_.data(), &l.sm, &l.mn);
    linear(wl.qc, l.xn2, rows_, &l.qc);
    linear(wl.kc, l.mn, mrows_, &l.kc);
    linear(wl.vc, l.mn, mrows_, &l.vc);
    zeroed(&l.ctx_c, rows_ * HD);
    l.pc.resize(co_[nb_]);
    l.r.assign(mrows_, T{});
    each_pass([&](std::size_t b) {
      const std::size_t nq = query_size(b), nm = memory_size(b);
      if (nm == 0) return;
      const std::size_t o = qo_[b] * HD, mo = mo_[b] * HD;
      T* p = l.pc.data() + co_[b];
      Attn::attend(l.qc.data() + o, l.kc.data() + mo, l.vc.data() + mo, nq, nm, p, l.ctx_c.data() + o);
      T* rb = l.r.data() + mo_[b];
      for (std::size_t hi = 0; hi < Heads * nq; ++hi) {
        for (std::size_t j = 0; j < nm; ++j) rb[j] += p[hi * nm + j];
      }
      for (std::size_t j = 0; j < nm; ++j) rb[j] = T{1} - std::exp(-rb[j] / static_cast<T>(Heads));
    });
    linear(wl.oc, l.ctx_c, rows_, &u_);
    l.x2 = l.x1;
    add_masked(&l.x2, u_, cmask_, rows_);
    // feed-forward
    norm(wl.n3, l.x2, rows_, qmask_.data(), &l.s3, &l.xn3);
    linear(wl.f1, l.xn3, rows_, &l.pre);
    l.act = l.pre;
    for (T& a : l.act) a = a > T{} ? a : T{};
    linear(wl.f2, l.act, rows_, &u_);
    l.x3 = l.x2;
    add_masked(&l.x3, u_, mask_, rows_);
  }

  // d_, d_m_ and dr_ hold the gradients on the level's query rows,
  // workspace rows and relevance, and become those on its inputs (dr_: on
  // the relevance it came in with). x, m: the level's inputs.
  void level_backward(WLayer& wl, WFocus& wf, Acts& l, const std::vector<T>& x, const std::vector<T>& m) {
    // feed-forward
    masked_hidden_grad(d_, mask_, rows_, &g_);
    linear_back(wl.f2, l.act, g_, rows_, &ga_);
    for (std::size_t i = 0; i < ga_.size(); ++i) {
      if (l.pre[i] <= T{}) ga_[i] = T{};
    }
    linear_back(wl.f1, l.xn3, ga_, rows_, &gs_);
    norm_back(wl.n3, l.x2, gs_, rows_, qmask_.data(), l.s3, &d_);
    // cross attention; the relevance's gradient goes onto its probabilities:
    // dL/dp_hij = dL/dr_j (1 - r_j) / Heads, for every head and query row
    if (mrows_ > 0) {
      e_.resize(mrows_);
      for (std::size_t j = 0; j < mrows_; ++j) e_[j] = dr_[j] * (T{1} - l.r[j]) / static_cast<T>(Heads);
      masked_hidden_grad(d_, cmask_, rows_, &g_);
      linear_back(wl.oc, l.ctx_c, g_, rows_, &gc_);
      zeroed(&gq_, rows_ * HD);
      zeroed(&gk_, mrows_ * HD);
      zeroed(&gv_, mrows_ * HD);
      each_pass([&](std::size_t b) {
        const std::size_t nm = memory_size(b);
        if (nm == 0) return;
        const std::size_t o = qo_[b] * HD, mo = mo_[b] * HD;
        Attn::attend_back(l.qc.data() + o, l.kc.data() + mo, l.vc.data() + mo, l.pc.data() + co_[b],
                          gc_.data() + o, query_size(b), nm, gq_.data() + o, gk_.data() + mo, gv_.data() + mo,
                          e_.data() + mo_[b]);
      });
      linear_back(wl.qc, l.xn2, gq_, rows_, &gs_);
      norm_back(wl.n2, l.x1, gs_, rows_, qmask_.data(), l.s2, &d_);
      linear_back(wl.kc, l.mn, gk_, mrows_, &gm_);
      linear_back_add(wl.vc, l.mn, gv_, mrows_, &gm_);
      norm_back(wl.nm, l.w1, gm_, mrows_, mmask_.data(), l.sm, &d_m_);
    }
    // query self attention
    masked_hidden_grad(d_, mask_, rows_, &g_);
    linear_back(wl.os, l.ctx, g_, rows_, &gc_);
    zeroed(&gq_, rows_ * HD);
    zeroed(&gk_, rows_ * HD);
    zeroed(&gv_, rows_ * HD);
    each_pass([&](std::size_t b) {
      const std::size_t o = qo_[b] * HD, n = query_size(b);
      Attn::attend_back(l.q.data() + o, l.k.data() + o, l.v.data() + o, l.ps.data() + so_[b], gc_.data() + o, n, n,
                        gq_.data() + o, gk_.data() + o, gv_.data() + o);
    });
    linear_back(wl.qs, l.xn1, gq_, rows_, &gs_);
    linear_back_add(wl.ks, l.xn1, gk_, rows_, &gs_);
    linear_back_add(wl.vs, l.xn1, gv_, rows_, &gs_);
    norm_back(wl.n1, x, gs_, rows_, qmask_.data(), l.s1, &d_);
    // workspace self attention: through the gate (into the relevance it
    // came in with and into O_w), and through the keys' bias log(r + eps)
    zeroed(&dr_, mrows_);
    if (mrows_ == 0) return;
    g_.resize(mrows_ * kHidden);
    for (std::size_t j = 0; j < mrows_; ++j) {
      const T* dj = d_m_.data() + j * kStream + kHiddenOffset;
      const T* mj = mwmask_.data() + j * kHidden;
      const T* uj = l.uw.data() + j * kHidden;
      T* gj = g_.data() + j * kHidden;
      for (std::size_t k = 0; k < kHidden; ++k) {
        dr_[j] += mj[k] * uj[k] * dj[k];
        gj[k] = l.rin[j] * mj[k] * dj[k];
      }
    }
    linear_back(wf.ow, l.ctxw, g_, mrows_, &gc_);
    zeroed(&gq_, mrows_ * HD);
    zeroed(&gk_, mrows_ * HD);
    zeroed(&gv_, mrows_ * HD);
    zeroed(&e_, mrows_);  // the bias's gradient
    each_pass([&](std::size_t b) {
      const std::size_t o = mo_[b] * HD, n = memory_size(b);
      if (n == 0) return;
      Attn::attend_back(l.qw.data() + o, l.kw.data() + o, l.vw.data() + o, l.pw.data() + wo_[b], gc_.data() + o, n,
                        n, gq_.data() + o, gk_.data() + o, gv_.data() + o, nullptr, e_.data() + mo_[b]);
    });
    for (std::size_t j = 0; j < mrows_; ++j) dr_[j] += e_[j] / static_cast<T>(l.rin[j] + kFocusEps);
    linear_back(wf.qw, l.wn, gq_, mrows_, &gm_);
    linear_back_add(wf.kw, l.wn, gk_, mrows_, &gm_);
    linear_back_add(wf.vw, l.wn, gv_, mrows_, &gm_);
    norm_back(wf.nw, m, gm_, mrows_, mmask_.data(), l.sw, &d_m_);
  }

  std::vector<Acts> acts_;
  ThreadPool* pool_ = nullptr;

  // The batch: pass b's query rows are [qo_[b], qo_[b+1]), its workspace
  // rows [mo_[b], mo_[b+1]); so_ / co_ / wo_ place its query self, cross
  // and workspace self probabilities.
  std::size_t nb_ = 0, rows_ = 0, mrows_ = 0;
  std::vector<std::size_t> qo_, mo_, so_, co_, wo_;
  std::vector<Input> queries_, memory_;
  std::vector<T> x0_, m0_, qmask_, mask_, cmask_, mmask_, mwmask_, ones_;
  const std::vector<T>* final_ = nullptr;
  const std::vector<T>* mfinal_ = nullptr;
  // Gradients: on the query rows, the workspace rows, the relevance, and a
  // pointer's share of the final workspace rows waiting for backward().
  std::vector<T> d_, d_m_, dr_, d_m_extra_;
  // Scratch, reused across calls.
  std::vector<T> bias_, u_, g_, ga_, gs_, gc_, gq_, gk_, gv_, gm_, e_, row_;
};

}  // namespace zf
