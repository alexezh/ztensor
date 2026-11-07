#pragma once

// LowResCross on the CPU (the facade is lowrescross.h): the activations
// of a batch, and the memories' cache, over the facade's weights.
//
// A batch's rows are stacked into one matrix, so every projection and
// feed-forward is one matrix product over the whole batch (the GEMM path of
// Linear), and attention runs per pass over that pass's rows only:
// passes never see each other. The row-wise work (layer norms) and the
// per-pass attention can run on a thread pool (set_pool); results do not
// depend on how many threads it has. None by default: those loops are tens
// of microseconds each, too short to pay for the round trip, and training
// splits a batch across threads by example instead.
//
// The memories' cache: a memory row's cross K and V depend on that row
// alone (the memory is fixed through the stack), so each row goes through
// the memory side once -- its row, norm statistics, every layer's norm, K
// and V -- when it is added, and rows come and go without touching the
// others.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "layers.h"
#include "tensor.h"
#include "thread_pool.h"
#include "lowrescross_layout.h"

namespace zf {

template <typename T, typename Layout, std::size_t Q, std::size_t M, std::size_t Heads,
          std::size_t HeadDim, std::size_t Ffn>
class LowResCrossCpu {
 public:
  static constexpr std::size_t HD = Heads * HeadDim;
  // The row, and the state the layers write.
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  static constexpr std::size_t kHiddenOffset = Layout::kStateOffset;
  using Input = typename Layout::Input;
  using StateGrad = typename Layout::StateGrad;
  using Pass = LowResPass<Input>;
  using Weights = LowResCrossWeights<T, Layout, HD, Ffn>;
  using WLayer = typename Weights::Layer;

  void set_pool(ThreadPool* pool) { pool_ = pool; }

  void forward_batch(Weights& w, const std::vector<Pass>& batch) {
    shared_ = false;
    pending_ = false;
    memory_.clear();
    mrows_ = 0;
    mb_.assign(batch.size(), 0);
    ml_.assign(batch.size(), 0);
    for (std::size_t b = 0; b < batch.size(); ++b) {
      mb_[b] = memory_.size();
      ml_[b] = batch[b].memory->size();
      memory_.insert(memory_.end(), batch[b].memory->begin(), batch[b].memory->end());
    }
    acts_.resize(w.layers.size());
    load_memory(w, 0);
    std::vector<const std::vector<Input>*> queries;
    for (const Pass& in : batch) queries.push_back(in.queries);
    run_passes(w, queries);
  }

  // --- the memories' cache ---

  void set_memories(Weights& w, const std::vector<const std::vector<Input>*>& memories) {
    clear_memories();
    add_memories(w, memories);
  }
  void clear_memories() {
    if (pending_) throw std::logic_error("LowResCross: memories changed before backward_memories()");
    shared_ = true;
    memory_.clear();
    mrows_ = 0;
    sb_.clear();
    sl_.clear();
    for (Acts& l : acts_) {
      l.gkm.clear();
      l.gvm.clear();
    }
  }
  std::size_t add_memory(Weights& w, const std::vector<Input>& rows) {
    add_memories(w, {&rows});
    return sb_.size() - 1;
  }
  // `rows` (indices into memory j, any order) out of memory j.
  void remove_memory_rows(std::size_t j, std::vector<std::size_t> rows) {
    if (!shared_ || j >= sb_.size()) throw std::out_of_range("LowResCross: no such memory");
    if (pending_) throw std::logic_error("LowResCross: memories changed before backward_memories()");
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (std::size_t a = rows.size(); a-- > 0;) {
      if (rows[a] >= sl_[j]) throw std::out_of_range("LowResCross: no such memory row");
      const std::size_t r = sb_[j] + rows[a];
      memory_.erase(memory_.begin() + static_cast<std::ptrdiff_t>(r));
      erase_row(&m_, r, kStream);
      erase_row(&mmask_, r, kStream);
      erase_row(&d_m_extra_, r, kStream);
      mstats_.erase(mstats_.begin() + static_cast<std::ptrdiff_t>(r));
      for (Acts& l : acts_) {
        erase_row(&l.mn, r, kStream);
        for (auto* v : {&l.kc_, &l.vc_, &l.gkm, &l.gvm}) erase_row(v, r, HD);
      }
    }
    mrows_ -= rows.size();
    sl_[j] -= rows.size();
    for (std::size_t k = j + 1; k < sb_.size(); ++k) sb_[k] -= rows.size();
  }
  std::size_t memory_count() const { return shared_ ? sb_.size() : 0; }

  void forward_shared(Weights& w, const std::vector<const std::vector<Input>*>& queries,
                      const std::vector<std::size_t>& memory_of) {
    if (!shared_) throw std::logic_error("LowResCross: forward_shared() without memories");
    mb_.assign(queries.size(), 0);
    ml_.assign(queries.size(), 0);
    for (std::size_t b = 0; b < queries.size(); ++b) {
      mb_[b] = sb_.at(memory_of[b]);
      ml_[b] = sl_.at(memory_of[b]);
    }
    run_passes(w, queries);
  }
  void backward_memories(Weights& w) {
    if (!shared_) throw std::logic_error("LowResCross: backward_memories() without memories");
    d_m_.swap(d_m_extra_);
    d_m_.resize(mrows_ * kStream, T{});
    for (std::size_t li = acts_.size(); li-- > 0;) {
      Acts& l = acts_[li];
      if (mrows_ > 0) memory_backward(w.layers[li], l, l.gkm, l.gvm);
      zeroed(&l.gkm, mrows_ * HD);
      zeroed(&l.gvm, mrows_ * HD);
    }
    for (std::size_t j = 0; j < mrows_; ++j) w.identity_backward(memory_[j], d_m_.data() + j * kStream);
    zeroed(&d_m_extra_, mrows_ * kStream);
    pending_ = false;
  }

  // --- the batch just run ---

  std::size_t batch_size() const { return nb_; }
  std::size_t query_size(std::size_t b) const { return qo_[b + 1] - qo_[b]; }
  // Pass b's query row i's final hidden state (kHidden).
  const T* hidden_row(std::size_t b, std::size_t i) const {
    return final_->data() + (qo_[b] + i) * kStream + kHiddenOffset;
  }
  std::size_t memory_size(std::size_t b) const { return ml_[b]; }
  const T* memory_rows(std::size_t b) const { return m_.data() + mb_[b] * kStream; }
  T* memory_grad(std::size_t b) { return d_m_extra_.data() + mb_[b] * kStream; }

  std::vector<std::vector<StateGrad>> backward_batch(Weights& w, const std::vector<Tensor<T, Q, kHidden>>& d_readout) {
    if (d_readout.size() != nb_) throw std::invalid_argument("LowResCross: one gradient per pass");
    zeroed(&d_, rows_ * kStream);
    for (std::size_t b = 0; b < nb_; ++b) {
      for (std::size_t i = 0; i < query_size(b); ++i) {
        T* d = d_.data() + (qo_[b] + i) * kStream + kHiddenOffset;
        for (std::size_t k = 0; k < kHidden; ++k) d[k] = d_readout[b](i, k);
      }
    }
    // Shared memories keep their gradients for backward_memories().
    if (shared_) {
      pending_ = true;
    } else {
      d_m_.swap(d_m_extra_);
    }
    for (std::size_t li = acts_.size(); li-- > 0;) {
      layer_backward(w.layers[li], acts_[li], li == 0 ? x0_ : acts_[li - 1].x3);
    }
    // Id embeddings: the learned id fields of every row.
    for (std::size_t i = 0; i < rows_; ++i) w.identity_backward(queries_[i], d_.data() + i * kStream);
    if (!shared_) {
      for (std::size_t j = 0; j < mrows_; ++j) w.identity_backward(memory_[j], d_m_.data() + j * kStream);
    }
    std::vector<std::vector<StateGrad>> out(nb_);
    for (std::size_t b = 0; b < nb_; ++b) {
      out[b].reserve(query_size(b));
      for (std::size_t i = 0; i < query_size(b); ++i) {
        const std::size_t row = qo_[b] + i;
        out[b].push_back(Layout::state_grad(d_.data() + row * kStream + kHiddenOffset, queries_[row]));
      }
    }
    if (!shared_) zeroed(&d_m_extra_, mrows_ * kStream);
    return out;
  }

 private:
  using RowStats = typename LayerNorm<T, kStream>::RowStats;

  // A layer's activations, row-stacked: rows_ query rows, mrows_ memory rows.
  struct Acts {
    std::vector<T> xn1, x1, xn2, x2, xn3, x3;  // rows x kStream
    std::vector<T> mn;                          // mrows x kStream
    std::vector<T> q, k, v, ctx, qc_, ctx_c;    // rows x HD
    std::vector<T> kc_, vc_;                    // mrows x HD
    std::vector<T> gkm, gvm;                    // shared memories' K, V gradients
    std::vector<T> ps, pc;                      // attention probabilities
    std::vector<T> pre, act;                    // rows x Ffn
    std::vector<RowStats> s1, s2, s3;           // n1, n2, n3's row statistics
  };

  static void zeroed(std::vector<T>* v, std::size_t n) { v->assign(n, T{}); }
  static void erase_row(std::vector<T>* v, std::size_t r, std::size_t width) {
    v->erase(v->begin() + static_cast<std::ptrdiff_t>(r * width),
             v->begin() + static_cast<std::ptrdiff_t>((r + 1) * width));
  }

  // The memories appended to the cache, their rows through the memory side
  // together.
  void add_memories(Weights& w, const std::vector<const std::vector<Input>*>& memories) {
    if (!shared_) clear_memories();
    if (pending_) throw std::logic_error("LowResCross: memories changed before backward_memories()");
    const std::size_t r0 = memory_.size();
    for (const auto* rows : memories) {
      sb_.push_back(memory_.size());
      sl_.push_back(rows->size());
      memory_.insert(memory_.end(), rows->begin(), rows->end());
    }
    acts_.resize(w.layers.size());
    load_memory(w, r0);
    for (std::size_t li = 0; li < acts_.size(); ++li) {
      memory_forward(w.layers[li], acts_[li], r0);
      acts_[li].gkm.resize(mrows_ * HD, T{});
      acts_[li].gvm.resize(mrows_ * HD, T{});
    }
  }

  // memory_'s rows from r0 on into m_ and their masks and norm statistics.
  void load_memory(const Weights& w, std::size_t r0) {
    mrows_ = memory_.size();
    m_.resize(r0 * kStream);
    mmask_.resize(r0 * kStream);
    m_.resize(mrows_ * kStream, T{});
    mmask_.resize(mrows_ * kStream, T{});
    for (std::size_t j = r0; j < mrows_; ++j) {
      w.put(m_.data() + j * kStream, memory_[j]);
      Layout::put_mask(mmask_.data() + j * kStream, memory_[j]);
    }
    // The memory rows are the same at every layer, and so are their norm
    // statistics (which do not depend on a norm's gamma and beta).
    mstats_.resize(mrows_);
    if (mrows_ > r0 && !w.layers.empty()) {
      w.layers[0].nm.stats_rows(m_.data() + r0 * kStream, mmask_.data() + r0 * kStream, mrows_ - r0,
                                mstats_.data() + r0, pool_);
    }
    if (r0 == 0) d_m_extra_.clear();
    d_m_extra_.resize(mrows_ * kStream, T{});
  }

  // The passes, over the memory already loaded (mb_ / ml_ place each pass's).
  void run_passes(Weights& w, const std::vector<const std::vector<Input>*>& queries) {
    nb_ = queries.size();
    qo_.assign(nb_ + 1, 0);
    so_.assign(nb_ + 1, 0);
    co_.assign(nb_ + 1, 0);
    for (std::size_t b = 0; b < nb_; ++b) {
      const std::size_t nq = queries[b]->size(), nm = ml_[b];
      if (nq > Q || nm > M) throw std::length_error("LowResCross overflow");
      qo_[b + 1] = qo_[b] + nq;
      so_[b + 1] = so_[b] + Heads * nq * nq;
      co_[b + 1] = co_[b] + Heads * nq * nm;
    }
    rows_ = qo_[nb_];
    queries_.clear();
    for (const auto* q : queries) queries_.insert(queries_.end(), q->begin(), q->end());
    zeroed(&x0_, rows_ * kStream);
    for (std::size_t i = 0; i < rows_; ++i) w.put(x0_.data() + i * kStream, queries_[i]);
    // Where a layer may write: the active regions, and for the cross
    // attention only in passes that have a memory.
    zeroed(&mask_, rows_ * kHidden);
    zeroed(&cmask_, rows_ * kHidden);
    // Read masks for the norms: which of each row's kStream slots are present.
    zeroed(&qmask_, rows_ * kStream);
    for (std::size_t i = 0; i < rows_; ++i) Layout::put_mask(qmask_.data() + i * kStream, queries_[i]);
    for (std::size_t b = 0; b < nb_; ++b) {
      for (std::size_t i = qo_[b]; i < qo_[b + 1]; ++i) {
        Layout::put_write_mask(mask_.data() + i * kHidden, queries_[i]);
        if (ml_[b] > 0) Layout::put_write_mask(cmask_.data() + i * kHidden, queries_[i]);
      }
    }
    const std::vector<T>* x = &x0_;
    for (std::size_t li = 0; li < acts_.size(); ++li) {
      layer_forward(w.layers[li], acts_[li], *x);
      x = &acts_[li].x3;
    }
    final_ = x;
  }

  // A layer's memory side for rows r0 on: the rows' norm, then the cross K
  // and V.
  void memory_forward(const WLayer& wl, Acts& l, std::size_t r0) {
    const std::size_t n = mrows_ - r0;
    l.mn.resize(mrows_ * kStream);
    l.kc_.resize(mrows_ * HD);
    l.vc_.resize(mrows_ * HD);
    if (n == 0) return;
    wl.nm.forward_rows_from(mstats_.data() + r0, m_.data() + r0 * kStream, l.mn.data() + r0 * kStream, n,
                            mmask_.data() + r0 * kStream, pool_);
    wl.kc.forward_into(l.mn.data() + r0 * kStream, l.kc_.data() + r0 * HD, n);
    wl.vc.forward_into(l.mn.data() + r0 * kStream, l.vc_.data() + r0 * HD, n);
  }
  // Its backward from the gradients on K and V; adds into d_m_.
  void memory_backward(WLayer& wl, Acts& l, const std::vector<T>& gk, const std::vector<T>& gv) {
    linear_back(wl.kc, l.mn, gk, mrows_, &gm_);
    linear_back_add(wl.vc, l.mn, gv, mrows_, &gm_);
    norm_back(wl.nm, m_, gm_, mrows_, mmask_.data(), mstats_, &d_m_);
  }

  // --- row-stacked pieces ---

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

  // The norm is masked by `mask` (rows x kStream): a token's absent regions
  // are excluded from the statistics and driven to zero, so an absent region
  // contributes nothing to the Q/K/V it feeds -- see LayerNorm masked path.
  // *stats gets the rows' statistics, which norm_back reuses.
  void norm(const LayerNorm<T, kStream>& n, const std::vector<T>& x, std::size_t rows, const T* mask,
            std::vector<RowStats>* stats, std::vector<T>* out) const {
    stats->resize(rows);
    n.stats_rows(x.data(), mask, rows, stats->data(), pool_);
    out->resize(rows * kStream);
    n.forward_rows_from(stats->data(), x.data(), out->data(), rows, mask, pool_);
  }
  // Adds dL/dx into *dx.
  void norm_back(LayerNorm<T, kStream>& n, const std::vector<T>& x, const std::vector<T>& d_out,
                 std::size_t rows, const T* mask, const std::vector<RowStats>& stats, std::vector<T>* dx) {
    row_.resize(rows * kStream);
    n.backward_rows_from(stats.data(), x.data(), d_out.data(), row_.data(), rows, mask, pool_);
    for (std::size_t i = 0; i < rows * kStream; ++i) (*dx)[i] += row_[i];
  }
  // f(b) for every pass of the batch, on the pool.
  template <typename F>
  void each_pass(F&& f) const {
    parallel_for(pool_, nb_, 1, [&](std::size_t, std::size_t b0, std::size_t b1) {
      for (std::size_t b = b0; b < b1; ++b) f(b);
    });
  }

  // x (rows x kStream) += mask · u (rows x kHidden).
  void add_masked(std::vector<T>* x, const std::vector<T>& u, const std::vector<T>& mask) const {
    for (std::size_t i = 0; i < rows_; ++i) {
      T* h = x->data() + i * kStream + kHiddenOffset;
      const T* ui = u.data() + i * kHidden;
      const T* mi = mask.data() + i * kHidden;
      for (std::size_t k = 0; k < kHidden; ++k) h[k] += mi[k] * ui[k];
    }
  }
  void masked_hidden_grad(const std::vector<T>& d, const std::vector<T>& mask, std::vector<T>* g) const {
    g->resize(rows_ * kHidden);
    for (std::size_t i = 0; i < rows_; ++i) {
      const T* di = d.data() + i * kStream + kHiddenOffset;
      const T* mi = mask.data() + i * kHidden;
      T* gi = g->data() + i * kHidden;
      for (std::size_t k = 0; k < kHidden; ++k) gi[k] = mi[k] * di[k];
    }
  }

  // One head's slice of two rows: HeadDim long, four lanes at a time.
  static T dot(const T* a, const T* b) {
    if constexpr (HeadDim % 4 == 0) {
      Vec4<T> acc{};
      for (std::size_t d = 0; d < HeadDim; d += 4) acc += load4(a + d) * load4(b + d);
      return hsum4(acc);
    } else {
      T s{};
      for (std::size_t d = 0; d < HeadDim; ++d) s += a[d] * b[d];
      return s;
    }
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

  // Scaled dot-product attention of nq query rows over nk key rows, rows of
  // HD (Heads x HeadDim). p: Heads x nq x nk. Overwrites ctx's nq rows.
  static void attend(const T* q, const T* k, const T* v, std::size_t nq, std::size_t nk, T* p, T* ctx) {
    const T scale = static_cast<T>(1.0 / std::sqrt(static_cast<double>(HeadDim)));
    if constexpr (HeadDim % 4 != 0) std::fill(ctx, ctx + nq * HD, T{});
    for (std::size_t h = 0; h < Heads; ++h) {
      const std::size_t o = h * HeadDim;
      for (std::size_t i = 0; i < nq; ++i) {
        T* pi = p + (h * nq + i) * nk;
        T top = std::numeric_limits<T>::lowest();
        if constexpr (HeadDim % 4 == 0) {
          const Slice qi = Slice::load(q + i * HD + o);
          for (std::size_t j = 0; j < nk; ++j) {
            pi[j] = qi.dot(k + j * HD + o) * scale;
            top = std::max(top, pi[j]);
          }
        } else {
          const T* qi = q + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) {
            pi[j] = dot(qi, k + j * HD + o) * scale;
            top = std::max(top, pi[j]);
          }
        }
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

  // Adds into d_q, d_k, d_v.
  static void attend_back(const T* q, const T* k, const T* v, const T* p, const T* d_ctx, std::size_t nq,
                          std::size_t nk, T* d_q, T* d_k, T* d_v) {
    const T scale = static_cast<T>(1.0 / std::sqrt(static_cast<double>(HeadDim)));
    std::array<T, (Q > M ? Q : M)> dp{};
    for (std::size_t h = 0; h < Heads; ++h) {
      const std::size_t o = h * HeadDim;
      for (std::size_t i = 0; i < nq; ++i) {
        const T* pi = p + (h * nq + i) * nk;
        T pdp{};
        if constexpr (HeadDim % 4 == 0) {
          const Slice gi = Slice::load(d_ctx + i * HD + o);
          for (std::size_t j = 0; j < nk; ++j) {
            dp[j] = gi.dot(v + j * HD + o);
            pdp += dp[j] * pi[j];
            gi.add_to(pi[j], d_v + j * HD + o);
          }
          const Slice qi = Slice::load(q + i * HD + o);
          Slice dqi = Slice::load(d_q + i * HD + o);
          for (std::size_t j = 0; j < nk; ++j) {
            const T ds = pi[j] * (dp[j] - pdp) * scale;
            dqi.add(ds, k + j * HD + o);
            qi.add_to(ds, d_k + j * HD + o);
          }
          dqi.store(d_q + i * HD + o);
        } else {
          const T* gi = d_ctx + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) {
            dp[j] = dot(gi, v + j * HD + o);
            pdp += dp[j] * pi[j];
            axpy(pi[j], gi, d_v + j * HD + o);
          }
          const T* qi = q + i * HD + o;
          T* dqi = d_q + i * HD + o;
          for (std::size_t j = 0; j < nk; ++j) {
            const T ds = pi[j] * (dp[j] - pdp) * scale;
            axpy(ds, k + j * HD + o, dqi);
            axpy(ds, qi, d_k + j * HD + o);
          }
        }
      }
    }
  }

  void layer_forward(const WLayer& wl, Acts& l, const std::vector<T>& x) {
    // self
    norm(wl.n1, x, rows_, qmask_.data(), &l.s1, &l.xn1);
    linear(wl.qs, l.xn1, rows_, &l.q);
    linear(wl.ks, l.xn1, rows_, &l.k);
    linear(wl.vs, l.xn1, rows_, &l.v);
    l.ctx.resize(rows_ * HD);
    l.ps.resize(so_[nb_]);
    each_pass([&](std::size_t b) {
      const std::size_t o = qo_[b] * HD, n = query_size(b);
      attend(l.q.data() + o, l.k.data() + o, l.v.data() + o, n, n, l.ps.data() + so_[b], l.ctx.data() + o);
    });
    linear(wl.os, l.ctx, rows_, &u_);
    l.x1 = x;
    add_masked(&l.x1, u_, mask_);
    // cross
    norm(wl.n2, l.x1, rows_, qmask_.data(), &l.s2, &l.xn2);
    l.x2 = l.x1;
    if (mrows_ > 0) {
      if (!shared_) memory_forward(wl, l, 0);
      linear(wl.qc, l.xn2, rows_, &l.qc_);
      zeroed(&l.ctx_c, rows_ * HD);
      l.pc.resize(co_[nb_]);
      each_pass([&](std::size_t b) {
        const std::size_t nm = memory_size(b);
        if (nm == 0) return;
        const std::size_t o = qo_[b] * HD, mo = mb_[b] * HD;
        attend(l.qc_.data() + o, l.kc_.data() + mo, l.vc_.data() + mo, query_size(b), nm, l.pc.data() + co_[b],
               l.ctx_c.data() + o);
      });
      linear(wl.oc, l.ctx_c, rows_, &u_);
      add_masked(&l.x2, u_, cmask_);
    }
    // feed-forward
    norm(wl.n3, l.x2, rows_, qmask_.data(), &l.s3, &l.xn3);
    linear(wl.f1, l.xn3, rows_, &l.pre);
    l.act = l.pre;
    for (T& a : l.act) a = a > T{} ? a : T{};
    linear(wl.f2, l.act, rows_, &u_);
    l.x3 = l.x2;
    add_masked(&l.x3, u_, mask_);
  }

  // d_ holds dL/d(layer output) and becomes dL/d(layer input); d_m_ gets
  // the memory rows' share. `x` is the layer's input.
  void layer_backward(WLayer& wl, Acts& l, const std::vector<T>& x) {
    // feed-forward
    masked_hidden_grad(d_, mask_, &g_);
    linear_back(wl.f2, l.act, g_, rows_, &ga_);
    for (std::size_t i = 0; i < ga_.size(); ++i) {
      if (l.pre[i] <= T{}) ga_[i] = T{};
    }
    linear_back(wl.f1, l.xn3, ga_, rows_, &gs_);
    norm_back(wl.n3, l.x2, gs_, rows_, qmask_.data(), l.s3, &d_);
    // cross
    if (mrows_ > 0) {
      masked_hidden_grad(d_, cmask_, &g_);
      linear_back(wl.oc, l.ctx_c, g_, rows_, &gc_);
      zeroed(&gq_, rows_ * HD);
      // Shared memories gather their K, V gradients over the passes.
      std::vector<T>& gk = shared_ ? l.gkm : gk_;
      std::vector<T>& gv = shared_ ? l.gvm : gv_;
      if (!shared_) {
        zeroed(&gk_, mrows_ * HD);
        zeroed(&gv_, mrows_ * HD);
      }
      const auto back = [&](std::size_t b) {
        const std::size_t nm = memory_size(b);
        if (nm == 0) return;
        const std::size_t o = qo_[b] * HD, mo = mb_[b] * HD;
        attend_back(l.qc_.data() + o, l.kc_.data() + mo, l.vc_.data() + mo, l.pc.data() + co_[b], gc_.data() + o,
                    query_size(b), nm, gq_.data() + o, gk.data() + mo, gv.data() + mo);
      };
      // Passes that share a memory add into the same rows: one at a time.
      if (shared_) {
        for (std::size_t b = 0; b < nb_; ++b) back(b);
      } else {
        each_pass(back);
      }
      linear_back(wl.qc, l.xn2, gq_, rows_, &gs_);
      norm_back(wl.n2, l.x1, gs_, rows_, qmask_.data(), l.s2, &d_);
      if (!shared_) memory_backward(wl, l, gk_, gv_);
    }
    // self
    masked_hidden_grad(d_, mask_, &g_);
    linear_back(wl.os, l.ctx, g_, rows_, &gc_);
    zeroed(&gq_, rows_ * HD);
    zeroed(&gk_, rows_ * HD);
    zeroed(&gv_, rows_ * HD);
    each_pass([&](std::size_t b) {
      const std::size_t o = qo_[b] * HD, n = query_size(b);
      attend_back(l.q.data() + o, l.k.data() + o, l.v.data() + o, l.ps.data() + so_[b], gc_.data() + o, n, n,
                  gq_.data() + o, gk_.data() + o, gv_.data() + o);
    });
    linear_back(wl.qs, l.xn1, gq_, rows_, &gs_);
    linear_back_add(wl.ks, l.xn1, gk_, rows_, &gs_);
    linear_back_add(wl.vs, l.xn1, gv_, rows_, &gs_);
    norm_back(wl.n1, x, gs_, rows_, qmask_.data(), l.s1, &d_);
  }

  std::vector<Acts> acts_;
  ThreadPool* pool_ = nullptr;

  // The batch: pass b's query rows are [qo_[b], qo_[b+1]), its memory rows
  // ml_[b] from mb_[b]; so_ / co_ place its self / cross probabilities.
  // The memories' cache (shared_): memory j's rows are sl_[j] from sb_[j];
  // pending_ while their gradients wait for backward_memories().
  std::size_t nb_ = 0, rows_ = 0, mrows_ = 0;
  std::vector<std::size_t> qo_, mb_, ml_, so_, co_, sb_, sl_;
  bool shared_ = false, pending_ = false;
  std::vector<Input> queries_, memory_;
  std::vector<T> x0_, m_, mask_, cmask_, qmask_, mmask_;
  std::vector<RowStats> mstats_;  // the memory rows' norm statistics
  const std::vector<T>* final_ = nullptr;
  // Gradients: on the query rows, on the memory rows, and the pointer's
  // share of the memory rows waiting for backward().
  std::vector<T> d_, d_m_, d_m_extra_;
  // Scratch, reused across calls.
  std::vector<T> u_, g_, ga_, gs_, gc_, gq_, gk_, gv_, gm_, row_;
};

}  // namespace zf
