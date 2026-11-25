#pragma once

// FocusCross on the GPU (the facade is focuscross.h), on ztensor's
// zf::apple graphs (focuscross_graph.h has the stack). Built only with
// ztensor's GPU target (ZTENSOR_APPLE_GPU).
//
// As LowResCrossApple, over the facade's weights (weighted.h): the runs read
// a copy of them and their gradients come back into the facade's. Two
// graphs, each made for a capacity on first use (passes round up to powers
// of two; the rest is padding):
//
//   batch  forward_batch: the stack, its readouts, final workspace rows and
//          relevance; backward_batch runs it again with the gradients
//   step   step_batch: the batch graph with the model's heads and loss on
//          top, forward, loss and every gradient in one run (one per kind
//          of heads)
//
// There is no memories' cache: the workspace depends on the pass's queries.

#if defined(ZTENSOR_APPLE_GPU)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "apple/focuscross_graph.h"
#include "apple/graph.h"
#include "apple/lowrescross_graph.h"
#include "apple/weighted.h"
#include "focuscross_weights.h"
#include "lowrescross_layout.h"

namespace zf {

template <typename T, typename Layout, std::size_t Q, std::size_t M, std::size_t Heads,
          std::size_t HeadDim, std::size_t Ffn>
class FocusCrossApple {
 public:
  static_assert(std::is_same_v<T, float>, "FocusCross runs on the GPU in float");
  static constexpr std::size_t HD = Heads * HeadDim;
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  static constexpr std::size_t kHiddenOffset = Layout::kStateOffset;
  using Input = typename Layout::Input;
  using StateGrad = typename Layout::StateGrad;
  using Pass = LowResPass<Input>;
  using Weights = FocusCrossWeights<T, Layout, HD, Ffn>;
  using Builder = FocusCrossGraph<T, Layout, Heads, HD>;
  using Node = zf::apple::Node;
  using Buffer = zf::apple::Buffer;
  using Bind = zf::apple::Graph::Bind;
  static constexpr int kQ = static_cast<int>(Q);
  static constexpr int kM = static_cast<int>(M);
  static constexpr int kS = static_cast<int>(kStream);
  static constexpr int kH = static_cast<int>(kHidden);

  void forward_batch(Weights& w, const std::vector<Pass>& batch) {
    mode_ = Mode::kBatch;
    BatchGraph& bg = batch_graph(w, batch.size());
    runs_.upload(w);
    pack(bg.blk, batch);
    std::vector<Bind> in, out{{bg.hidden, bg.hidden_out}, {bg.res.rows, bg.rows_out}};
    for (std::size_t l = 0; l < bg.res.relevance.size(); ++l) out.push_back({bg.res.relevance[l], bg.rel_out[l]});
    bind_inputs(bg.g, bg.blk.in, &in);
    bg.g.g.run(in, out);
    hidden_ = bg.hidden_out;
    rows_ = bg.rows_out;
    rel_ = bg.rel_out;
    d_rows_ = bg.blk.in[bg.d_rows];
  }

  // A whole step's gradients in one run: the stack, `heads` on its readouts
  // and final workspace rows, their loss, and the gradients of every weight.
  // Nothing of the batch is left to read but what the heads read back.
  template <typename HeadSet>
  void step_batch(Weights& w, const std::vector<Pass>& batch, HeadSet& heads) {
    mode_ = Mode::kStep;
    StepGraph<HeadSet>& sg = step_graph(w, batch.size(), heads);
    runs_.upload(w);
    sg.heads.upload(heads);  // small: every run
    pack(sg.blk, batch);
    heads.fill(sg.nodes, sg.blk.in);
    std::vector<Bind> in, out;
    bind_inputs(sg.g, sg.blk.in, &in);
    sg.heads.bind(&in);
    sg.out.bind(&out);
    for (std::size_t i = 0; i < sg.head_grads.size(); ++i) out.push_back({sg.head_grads[i], sg.head_grad_bufs[i]});
    runs_.run_with_grads(sg.g, &in, &out);
    runs_.flush(w);
    zf::apple::add_grads(sg.head_grad_bufs, heads);
    heads.read(sg.nodes, sg.out);
  }

  // --- the batch just run ---

  std::size_t batch_size() const { return qo_.size() - 1; }
  std::size_t query_size(std::size_t b) const { return qo_[b + 1] - qo_[b]; }
  const T* hidden_row(std::size_t b, std::size_t i) const { return hidden_.data() + (b * Q + i) * kHidden; }
  std::size_t memory_size(std::size_t b) const { return ml_[b]; }
  const T* memory_rows(std::size_t b) const { return rows_.data() + b * M * kStream; }
  T* memory_grad(std::size_t b) { return d_rows_.data() + b * M * kStream; }
  const T* relevance(std::size_t b, std::size_t level) const { return rel_.at(level).data() + b * M; }

  std::vector<std::vector<StateGrad>> backward_batch(Weights& w, const std::vector<zf::Tensor<T, Q, kHidden>>& d_readout) {
    if (mode_ == Mode::kStep) throw std::logic_error("FocusCross: no backward_batch() after step_batch()");
    if (d_readout.size() != batch_size()) throw std::invalid_argument("FocusCross: one gradient per pass");
    BatchGraph& bg = *batch_graphs_[cap_b_];
    std::vector<Bind> in, out{{bg.dx, bg.dx_out}};
    bind_inputs(bg.g, bg.blk.in, &in);
    Buffer d_hidden = bg.blk.in[bg.d_hidden];
    d_hidden.zero();
    for (std::size_t b = 0; b < d_readout.size(); ++b) {
      for (std::size_t i = 0; i < query_size(b); ++i) {
        float* d = d_hidden.data() + (b * Q + i) * kHidden;
        for (std::size_t k = 0; k < kHidden; ++k) d[k] = d_readout[b](i, k);
      }
    }
    runs_.run_with_grads(bg.g, &in, &out);
    runs_.flush(w);
    bg.blk.in[bg.d_rows].zero();
    std::vector<std::vector<StateGrad>> grads(batch_size());
    for (std::size_t b = 0; b < batch_size(); ++b) {
      grads[b].reserve(query_size(b));
      for (std::size_t i = 0; i < query_size(b); ++i) {
        const float* d = bg.dx_out.data() + (b * Q + i) * kStream + kHiddenOffset;
        grads[b].push_back(Layout::state_grad(d, queries_[qo_[b] + i]));
      }
    }
    return grads;
  }

 private:
  enum class Mode { kBatch, kStep };
  using Weighted = zf::apple::Weighted;

  // Passes as the graphs take them: each pass's query rows, its workspace
  // rows, where those may write and which are rows; and the stack over them.
  struct Block {
    int batch = 0;
    LowResGraphInputs in;
    typename Builder::Queries q;
    typename Builder::Rows m;
    Node wmask, cbias;
  };
  struct BatchGraph {
    Weighted g;
    Block blk;
    typename Builder::Result res;
    Node d_hidden, d_rows, hidden, dx;
    Buffer hidden_out, rows_out, dx_out;
    std::vector<Buffer> rel_out;
  };
  struct AnyStep {
    virtual ~AnyStep() = default;
  };
  template <typename HeadSet>
  struct StepGraph : AnyStep {
    Weighted g;
    Block blk;
    zf::apple::Params heads;
    std::vector<Node> head_grads;
    std::vector<Buffer> head_grad_bufs;
    LowResGraphOutputs out;
    typename HeadSet::Nodes nodes;
  };

  static int capacity(std::size_t n) {
    int c = 8;
    while (static_cast<std::size_t>(c) < n) c *= 2;
    return c;
  }
  static Node dot_sum(zf::apple::Graph& g, Node a, Node b) { return g.sum_all(g.mul(a, b)); }

  // A block of `batch` passes, and the stack over it.
  static typename Builder::Result make_block(Builder& net, zf::apple::Graph& g, int batch, Block* b) {
    b->batch = batch;
    b->q = net.queries(b->in, batch, kQ);
    b->m = net.rows(b->in, batch * kM);
    b->wmask = b->in.add(g, {batch * kM, kH});
    b->cbias = b->in.add(g, {batch, 1, 1, kM});
    return net.stack(b->q, b->m, b->wmask, b->cbias, kM);
  }

  BatchGraph& batch_graph(Weights& w, std::size_t n) {
    cap_b_ = capacity(n);
    auto& slot = batch_graphs_[cap_b_];
    if (slot != nullptr) return *slot;
    slot = std::make_unique<BatchGraph>();
    BatchGraph& bg = *slot;
    bg.g.declare(w);
    zf::apple::Graph& g = bg.g.g;
    Builder net(g, bg.g.params, w.layers.size());
    bg.res = make_block(net, g, cap_b_, &bg.blk);
    bg.hidden = net.hidden(bg.res.final_rows);
    bg.d_hidden = bg.blk.in.add(g, {cap_b_ * kQ, kH});
    bg.d_rows = bg.blk.in.add(g, {cap_b_ * kM, kS});
    const Node loss = g.add(dot_sum(g, bg.hidden, bg.d_hidden), dot_sum(g, bg.res.rows, bg.d_rows));
    bg.dx = bg.g.weigh(loss, {bg.blk.q.rows.x})[0];
    bg.hidden_out = Buffer(g.elements(bg.hidden));
    bg.rows_out = Buffer(g.elements(bg.res.rows));
    bg.dx_out = Buffer(g.elements(bg.dx));
    for (const Node r : bg.res.relevance) bg.rel_out.emplace_back(g.elements(r));
    return bg;
  }

  template <typename HeadSet>
  StepGraph<HeadSet>& step_graph(Weights& w, std::size_t n, HeadSet& heads) {
    cap_b_ = capacity(n);
    auto& slot = step_graphs_[{std::type_index(typeid(HeadSet)), cap_b_}];
    if (slot != nullptr) return static_cast<StepGraph<HeadSet>&>(*slot);
    auto made = std::make_unique<StepGraph<HeadSet>>();
    StepGraph<HeadSet>& sg = *made;
    sg.g.declare(w);
    zf::apple::Graph& g = sg.g.g;
    Builder net(g, sg.g.params, w.layers.size());
    const typename Builder::Result res = make_block(net, g, cap_b_, &sg.blk);
    sg.heads.declare(g, heads);
    LowResHeadsGraph hg{g, sg.heads, sg.blk.in, sg.out, cap_b_, kQ, kM, net.hidden(res.final_rows), res.rows};
    const Node loss = heads.build(hg, &sg.nodes);
    if (!sg.heads.all_taken()) throw std::logic_error("FocusCross: heads' parameters left over");
    sg.head_grads = sg.g.weigh(loss, sg.heads.nodes());
    for (const Node d : sg.head_grads) sg.head_grad_bufs.emplace_back(g.elements(d));
    slot = std::move(made);
    return sg;
  }

  void bind_inputs(const Weighted& wg, const LowResGraphInputs& gi, std::vector<Bind>* in) const {
    runs_.bind(wg, in);
    gi.bind(in);
  }

  // `batch` into a block's inputs (every input zeroed first), and the
  // batch's bookkeeping.
  void pack(Block& blk, const std::vector<Pass>& batch) {
    blk.in.zero();
    Builder::clear_queries(blk.in, blk.q);
    Builder::clear_ids(blk.in, blk.m);
    float* cbias = blk.in[blk.cbias].data();
    std::fill(cbias, cbias + blk.batch * kM, Builder::kNoKey);
    float* wmask = blk.in[blk.wmask].data();
    qo_.assign(1, 0);
    ml_.clear();
    queries_.clear();
    for (std::size_t b = 0; b < batch.size(); ++b) {
      const std::vector<Input>& ms = *batch[b].memory;
      if (ms.size() > M) throw std::length_error("FocusCross overflow");
      Builder::put_queries(blk.in, blk.q, static_cast<int>(b), *batch[b].queries, !ms.empty());
      for (std::size_t j = 0; j < ms.size(); ++j) {
        const int row = static_cast<int>(b) * kM + static_cast<int>(j);
        Builder::put_row(blk.in, blk.m, row, ms[j]);
        Layout::put_write_mask(wmask + static_cast<std::size_t>(row) * kHidden, ms[j]);
        cbias[row] = 0.0f;
      }
      queries_.insert(queries_.end(), batch[b].queries->begin(), batch[b].queries->end());
      qo_.push_back(queries_.size());
      ml_.push_back(ms.size());
    }
  }

  Mode mode_ = Mode::kBatch;
  int cap_b_ = 0;
  std::map<int, std::unique_ptr<BatchGraph>> batch_graphs_;
  std::map<std::pair<std::type_index, int>, std::unique_ptr<AnyStep>> step_graphs_;
  zf::apple::WeightRuns runs_;

  // The batch just run: pass b's queries are [qo_[b], qo_[b+1]) of
  // queries_, its workspace rows ml_[b] from b * M of rows_.
  std::vector<std::size_t> qo_{0}, ml_;
  std::vector<Input> queries_;
  Buffer hidden_, rows_, d_rows_;
  std::vector<Buffer> rel_;
};

}  // namespace zf

#endif  // ZTENSOR_APPLE_GPU
