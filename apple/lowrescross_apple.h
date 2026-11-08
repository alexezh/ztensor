#pragma once

// LowResCross on the GPU (the facade is lowrescross.h), on ztensor's
// zf::apple graphs (lowrescross_graph.h has the pieces). Built only with
// ztensor's GPU target (ZTENSOR_APPLE_GPU).
//
// The CPU model stays the master copy: the runs read a copy of the facade's
// weights, taken again only when their version has moved (once a step),
// and the gradients come back into the facade's, so the step stays on the
// CPU, the same for both implementations. The gradients add up on the GPU
// over a sequence of runs (each run adds its own into a running sum) and
// come back once at its end: after a batch's backward, or at
// backward_memories(). A backward run recomputes its forward on the GPU: a
// run keeps no activations.
//
// Four graphs, each made for a capacity on first use (passes and cache
// rows round up to powers of two; the rest is padding):
//
//   batch   forward_batch: each pass's memory side and the stack, in one
//   step    step_batch: the batch graph with the model's heads and loss on
//           top -- forward, loss and every gradient in one run, where
//           forward_batch, the heads on the CPU and backward_batch take
//           two runs and a wait between them (one per kind of heads)
//   memory  the memories' cache: rows in, every layer's cross K and V out
//   pass    forward_shared: the stack over K and V gathered from the cache
//
// The cache lives on the GPU: each row's K and V, made once when the row is
// added; removing rows closes the gap. The passes' gradients on K and V add
// up on the GPU (each pass run adds its own into a running sum) until
// backward_memories() takes the sum back through the memory side in one run.
// K and V are made from the weights of one version: a pass after the
// weights have moved makes them again, every row in one run.

#if defined(ZTENSOR_APPLE_GPU)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "apple/graph.h"
#include "apple/nn.h"
#include "apple/params.h"
#include "apple/lowrescross_graph.h"
#include "lowrescross_layout.h"

namespace zf {

template <typename T, typename Layout, std::size_t Q, std::size_t M, std::size_t Heads,
          std::size_t HeadDim, std::size_t Ffn>
class LowResCrossApple {
 public:
  static_assert(std::is_same_v<T, float>, "LowResCross runs on the GPU in float");
  static constexpr std::size_t HD = Heads * HeadDim;
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  static constexpr std::size_t kHiddenOffset = Layout::kStateOffset;
  using Input = typename Layout::Input;
  using StateGrad = typename Layout::StateGrad;
  using Pass = LowResPass<Input>;
  using Weights = LowResCrossWeights<T, Layout, HD, Ffn>;
  using Builder = LowResCrossGraph<T, Layout, Heads, HD>;
  using Node = zf::apple::Node;
  using Buffer = zf::apple::Buffer;
  using Bind = zf::apple::Graph::Bind;
  static constexpr int kQ = static_cast<int>(Q);
  static constexpr int kM = static_cast<int>(M);
  static constexpr int kS = static_cast<int>(kStream);
  static constexpr int kH = static_cast<int>(kHidden);
  static constexpr int kHD = static_cast<int>(HD);

  void forward_batch(Weights& w, const std::vector<Pass>& batch) {
    mode_ = Mode::kBatch;
    BatchGraph& bg = batch_graph(w, batch.size());
    upload(w);
    pack(bg.blk, batch);
    std::vector<Bind> in, out{{bg.hidden, bg.hidden_out}, {bg.blk.m.rows, bg.rows_out}};
    bind_inputs(bg.g, bg.blk.in, &in);
    bg.g.g.run(in, out);
    hidden_ = bg.hidden_out;
    rows_ = bg.rows_out;
    d_rows_ = bg.blk.in[bg.d_rows];
  }

  // A whole step's gradients in one run: the stack, `heads` on its readouts
  // and memory rows, their loss, and the gradients of every weight -- the
  // stack's into the facade's, the heads' into theirs. Nothing of the batch
  // is left to read but what the heads read back (no readouts, no
  // backward_batch()).
  template <typename HeadSet>
  void step_batch(Weights& w, const std::vector<Pass>& batch, HeadSet& heads) {
    mode_ = Mode::kStep;
    StepGraph<HeadSet>& sg = step_graph(w, batch.size(), heads);
    upload(w);
    sg.heads.upload(heads);  // small: every run
    pack(sg.blk, batch);
    heads.fill(sg.nodes, sg.blk.in);
    std::vector<Bind> in, out;
    bind_inputs(sg.g, sg.blk.in, &in);
    sg.heads.bind(&in);
    sg.out.bind(&out);
    for (std::size_t i = 0; i < sg.head_grads.size(); ++i) out.push_back({sg.head_grads[i], sg.head_grad_bufs[i]});
    run_with_grads(sg.g, &in, &out);
    flush_grads(w);
    zf::apple::add_grads(sg.head_grad_bufs, heads);
    heads.read(sg.nodes, sg.out);
  }

  // --- the memories' cache ---

  void set_memories(Weights& w, const std::vector<const std::vector<Input>*>& memories) {
    clear_memories();
    add_memories(w, memories);
  }
  void clear_memories() {
    if (pending_) throw std::logic_error("LowResCross: memories changed before backward_memories()");
    memory_.clear();
    sb_.clear();
    sl_.clear();
  }
  std::size_t add_memory(Weights& w, const std::vector<Input>& rows) {
    add_memories(w, {&rows});
    return sb_.size() - 1;
  }
  void remove_memory_rows(std::size_t j, std::vector<std::size_t> rows) {
    if (j >= sb_.size()) throw std::out_of_range("LowResCross: no such memory");
    if (pending_) throw std::logic_error("LowResCross: memories changed before backward_memories()");
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (std::size_t a = rows.size(); a-- > 0;) {
      if (rows[a] >= sl_[j]) throw std::out_of_range("LowResCross: no such memory row");
      const std::size_t r = sb_[j] + rows[a], n = memory_.size();
      memory_.erase(memory_.begin() + static_cast<std::ptrdiff_t>(r));
      cache_.erase_row(r, n);
    }
    sl_[j] -= rows.size();
    for (std::size_t k = j + 1; k < sb_.size(); ++k) sb_[k] -= rows.size();
  }
  std::size_t memory_count() const { return sb_.size(); }

  void forward_shared(Weights& w, const std::vector<const std::vector<Input>*>& queries,
                      const std::vector<std::size_t>& memory_of) {
    mode_ = Mode::kShared;
    PassGraph& pg = pass_graph(w, queries.size());
    upload(w);
    pg.in.zero();
    Builder::clear_queries(pg.in, pg.q);
    float* cbias = pg.in[pg.cbias].data();
    std::fill(cbias, cbias + pg.batch * kM, Builder::kNoKey);
    std::int32_t* idx = pg.in[pg.idx].ints();
    begin_batch(queries.size());
    for (std::size_t b = 0; b < queries.size(); ++b) {
      const std::size_t j = memory_of[b];
      if (j >= sb_.size()) throw std::out_of_range("LowResCross: no such memory");
      if (sl_[j] > M) throw std::length_error("LowResCross overflow");
      Builder::put_queries(pg.in, pg.q, static_cast<int>(b), *queries[b], sl_[j] > 0);
      for (std::size_t t = 0; t < sl_[j]; ++t) {
        idx[b * M + t] = static_cast<std::int32_t>(sb_[j] + t);
        cbias[b * M + t] = 0.0f;
      }
      add_pass(*queries[b], sb_[j], sl_[j]);
    }
    if (kv_version_ != w.version) encode_memories(w, 0);
    std::vector<Bind> in, out{{pg.hidden, pg.hidden_out}};
    bind_inputs(pg.g, pg.in, &in);
    cache_.bind_kv(pg.k, pg.v, &in);
    pg.g.g.run(in, out);
    hidden_ = pg.hidden_out;
    rows_ = cache_.rows;
    d_rows_ = cache_.d_rows;
  }

  void backward_memories(Weights& w) {
    MemoryGraph& mg = memory_graph(w);
    mg.in.zero();
    Builder::clear_ids(mg.in, mg.m);
    for (std::size_t r = 0; r < memory_.size(); ++r) Builder::put_row(mg.in, mg.m, static_cast<int>(r), memory_[r]);
    std::vector<Bind> in, out;
    bind_inputs(mg.g, mg.in, &in);
    for (std::size_t l = 0; l < mg.dk.size(); ++l) {
      in.push_back({mg.dk[l], cache_.gk[l]});
      in.push_back({mg.dv[l], cache_.gv[l]});
    }
    in.push_back({mg.d_rows, cache_.d_rows});
    run_with_grads(mg.g, &in, &out);
    flush_grads(w);
    cache_.zero_grads();
    pending_ = false;
  }

  // --- the batch just run ---

  std::size_t batch_size() const { return qo_.size() - 1; }
  std::size_t query_size(std::size_t b) const { return qo_[b + 1] - qo_[b]; }
  const T* hidden_row(std::size_t b, std::size_t i) const {
    return hidden_.data() + (b * Q + i) * kHidden;
  }
  std::size_t memory_size(std::size_t b) const { return ml_[b]; }
  const T* memory_rows(std::size_t b) const { return rows_.data() + mb_[b] * kStream; }
  T* memory_grad(std::size_t b) { return d_rows_.data() + mb_[b] * kStream; }

  std::vector<std::vector<StateGrad>> backward_batch(Weights& w, const std::vector<zf::Tensor<T, Q, kHidden>>& d_readout) {
    if (mode_ == Mode::kStep) throw std::logic_error("LowResCross: no backward_batch() after step_batch()");
    if (d_readout.size() != batch_size()) throw std::invalid_argument("LowResCross: one gradient per pass");
    std::vector<Bind> in, out;
    Buffer d_hidden, dx;
    if (mode_ == Mode::kBatch) {
      BatchGraph& bg = *batch_graphs_[cap_b_];
      bind_inputs(bg.g, bg.blk.in, &in);
      d_hidden = bg.blk.in[bg.d_hidden];
      dx = bg.dx_out;
      out.push_back({bg.dx, dx});
      fill_d_hidden(d_readout, &d_hidden);
      run_with_grads(bg.g, &in, &out);
      flush_grads(w);
      bg.blk.in[bg.d_rows].zero();
    } else {
      PassGraph& pg = *pass_graphs_[{cap_b_, cache_.cap}];
      bind_inputs(pg.g, pg.in, &in);
      cache_.bind_kv(pg.k, pg.v, &in);
      cache_.bind_sums(pg.gk_in, pg.gv_in, pg.gk_out, pg.gv_out, &in, &out);
      d_hidden = pg.in[pg.d_hidden];
      dx = pg.dx_out;
      out.push_back({pg.dx, dx});
      fill_d_hidden(d_readout, &d_hidden);
      run_with_grads(pg.g, &in, &out);
      cache_.flip_sums();
      pending_ = true;
    }
    std::vector<std::vector<StateGrad>> grads(batch_size());
    for (std::size_t b = 0; b < batch_size(); ++b) {
      grads[b].reserve(query_size(b));
      for (std::size_t i = 0; i < query_size(b); ++i) {
        const float* d = dx.data() + (b * Q + i) * kStream + kHiddenOffset;
        grads[b].push_back(Layout::state_grad(d, queries_[qo_[b] + i]));
      }
    }
    return grads;
  }

 private:
  enum class Mode { kBatch, kShared, kStep };

  // A graph with the weights as its inputs and their gradients: alone, and
  // added to a running sum (sum_in + grad).
  struct Weighted {
    zf::apple::Graph g;
    zf::apple::Params params;
    std::vector<Node> grads, sum_in, sum_out;
  };
  // Passes as the batch and step graphs take them: each pass's query rows,
  // its memory rows, and which of those are keys.
  struct Block {
    int batch = 0;
    LowResGraphInputs in;
    typename Builder::Queries q;
    typename Builder::Rows m;
    Node cbias;
  };
  struct BatchGraph {
    Weighted g;
    Block blk;
    Node d_hidden, d_rows, hidden, dx;
    Buffer hidden_out, rows_out, dx_out;
  };
  // One per kind of heads and capacity.
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
  struct MemoryGraph {
    int cap = 0;
    Weighted g;
    LowResGraphInputs in;
    typename Builder::Rows m;
    std::vector<Node> k, v, dk, dv;
    Node d_rows;
    Buffer rows_out;
    std::vector<Buffer> k_out, v_out;
  };
  struct PassGraph {
    int batch = 0;
    Weighted g;
    LowResGraphInputs in;
    typename Builder::Queries q;
    Node cbias, idx, d_hidden, hidden, dx;
    std::vector<Node> k, v, gk_in, gv_in, gk_out, gv_out;
    Buffer hidden_out, dx_out;
  };

  // The cache: cap rows of memory, each row's K and V for every layer, the
  // running sums of their gradients (two sets: a pass run reads one and
  // writes the other) and the pointer's gradient on the rows.
  struct Cache {
    int cap = 0;
    Buffer rows, d_rows;
    std::vector<Buffer> k, v, gk, gv, gk_next, gv_next;

    void grow(int to, std::size_t used, std::size_t levels) {
      Cache c;
      c.cap = to;
      c.rows = Buffer(static_cast<std::size_t>(to) * kStream);
      c.d_rows = Buffer(static_cast<std::size_t>(to) * kStream);
      for (std::size_t l = 0; l < levels; ++l) {
        for (auto* v : {&c.k, &c.v, &c.gk, &c.gv, &c.gk_next, &c.gv_next}) v->emplace_back(static_cast<std::size_t>(to) * HD);
      }
      if (cap > 0) {
        copy_rows(rows, c.rows, used, kStream);
        copy_rows(d_rows, c.d_rows, used, kStream);
        for (std::size_t l = 0; l < levels; ++l) {
          copy_rows(k[l], c.k[l], used, HD);
          copy_rows(v[l], c.v[l], used, HD);
          copy_rows(gk[l], c.gk[l], used, HD);
          copy_rows(gv[l], c.gv[l], used, HD);
        }
      }
      *this = std::move(c);
    }
    static void copy_rows(const Buffer& from, Buffer& to, std::size_t n, std::size_t width) {
      std::memcpy(to.data(), from.data(), n * width * sizeof(float));
    }
    // Row r of n out; the rows after it move up.
    void erase_row(std::size_t r, std::size_t n) {
      const auto close = [&](Buffer& b, std::size_t width) {
        float* p = b.data();
        std::memmove(p + r * width, p + (r + 1) * width, (n - r - 1) * width * sizeof(float));
        std::fill(p + (n - 1) * width, p + n * width, 0.0f);
      };
      close(rows, kStream);
      close(d_rows, kStream);
      for (auto* set : {&k, &v, &gk, &gv}) {
        for (Buffer& b : *set) close(b, HD);
      }
    }
    void bind_kv(const std::vector<Node>& kn, const std::vector<Node>& vn, std::vector<Bind>* in) const {
      for (std::size_t l = 0; l < kn.size(); ++l) {
        in->push_back({kn[l], k[l]});
        in->push_back({vn[l], v[l]});
      }
    }
    void bind_sums(const std::vector<Node>& gk_in, const std::vector<Node>& gv_in, const std::vector<Node>& gk_out,
                   const std::vector<Node>& gv_out, std::vector<Bind>* in, std::vector<Bind>* out) const {
      for (std::size_t l = 0; l < gk_in.size(); ++l) {
        in->push_back({gk_in[l], gk[l]});
        in->push_back({gv_in[l], gv[l]});
        out->push_back({gk_out[l], gk_next[l]});
        out->push_back({gv_out[l], gv_next[l]});
      }
    }
    void flip_sums() {
      gk.swap(gk_next);
      gv.swap(gv_next);
    }
    void zero_grads() {
      d_rows.zero();
      for (auto* set : {&gk, &gv}) {
        for (Buffer& b : *set) b.zero();
      }
    }
  };

  static int capacity(std::size_t n, int least) {
    int c = least;
    while (static_cast<std::size_t>(c) < n) c *= 2;
    return c;
  }

  void make_weighted(Weights& w, Weighted* out) {
    out->params.declare(out->g, w);
  }
  // The gradients of `loss` on the weights (and on `extra`, returned).
  std::vector<Node> weigh(Weighted& wg, Node loss, const std::vector<Node>& extra) {
    std::vector<Node> wrt = wg.params.nodes();
    wrt.insert(wrt.end(), extra.begin(), extra.end());
    std::vector<Node> all = wg.g.gradients(loss, wrt);
    wg.grads.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(wg.params.size()));
    for (const Node grad : wg.grads) {
      wg.sum_in.push_back(wg.g.input(wg.g.shape(grad)));
      wg.sum_out.push_back(wg.g.add(wg.sum_in.back(), grad));
    }
    return std::vector<Node>(all.begin() + static_cast<std::ptrdiff_t>(wg.params.size()), all.end());
  }
  static Node dot_sum(zf::apple::Graph& g, Node a, Node b) { return g.sum_all(g.mul(a, b)); }

  void make_block(Builder& net, zf::apple::Graph& g, int batch, Block* b) {
    b->batch = batch;
    b->q = net.queries(b->in, batch, kQ);
    b->m = net.rows(b->in, batch * kM);
    b->cbias = b->in.add(g, {batch, 1, 1, kM});
  }
  // The stack over a block's passes: its final hidden state, [batch * Q, kHidden].
  static Node block_hidden(Builder& net, const Block& b) {
    return net.hidden(net.stack(b.q, net.memory_side(b.m.rows, b.m.mask), b.cbias, kM));
  }

  BatchGraph& batch_graph(Weights& w, std::size_t n) {
    cap_b_ = capacity(n, 8);
    auto& slot = batch_graphs_[cap_b_];
    if (slot != nullptr) return *slot;
    slot = std::make_unique<BatchGraph>();
    BatchGraph& bg = *slot;
    make_weighted(w, &bg.g);
    zf::apple::Graph& g = bg.g.g;
    Builder net(g, bg.g.params, w.layers.size());
    make_block(net, g, cap_b_, &bg.blk);
    bg.d_hidden = bg.blk.in.add(g, {cap_b_ * kQ, kH});
    bg.d_rows = bg.blk.in.add(g, {cap_b_ * kM, kS});
    bg.hidden = block_hidden(net, bg.blk);
    const Node loss = g.add(dot_sum(g, bg.hidden, bg.d_hidden), dot_sum(g, bg.blk.m.rows, bg.d_rows));
    bg.dx = weigh(bg.g, loss, {bg.blk.q.rows.x})[0];
    bg.hidden_out = Buffer(g.elements(bg.hidden));
    bg.rows_out = Buffer(g.elements(bg.blk.m.rows));
    bg.dx_out = Buffer(g.elements(bg.dx));
    return bg;
  }

  template <typename HeadSet>
  StepGraph<HeadSet>& step_graph(Weights& w, std::size_t n, HeadSet& heads) {
    cap_b_ = capacity(n, 8);
    auto& slot = step_graphs_[{std::type_index(typeid(HeadSet)), cap_b_}];
    if (slot != nullptr) return static_cast<StepGraph<HeadSet>&>(*slot);
    auto made = std::make_unique<StepGraph<HeadSet>>();
    StepGraph<HeadSet>& sg = *made;
    make_weighted(w, &sg.g);
    zf::apple::Graph& g = sg.g.g;
    Builder net(g, sg.g.params, w.layers.size());
    make_block(net, g, cap_b_, &sg.blk);
    sg.heads.declare(g, heads);
    LowResHeadsGraph hg{g, sg.heads, sg.blk.in, sg.out, cap_b_, kQ, kM, block_hidden(net, sg.blk), sg.blk.m.rows};
    const Node loss = heads.build(hg, &sg.nodes);
    if (!sg.heads.all_taken()) throw std::logic_error("LowResCross: heads' parameters left over");
    sg.head_grads = weigh(sg.g, loss, sg.heads.nodes());
    for (const Node d : sg.head_grads) sg.head_grad_bufs.emplace_back(g.elements(d));
    slot = std::move(made);
    return sg;
  }

  MemoryGraph& memory_graph(Weights& w) {
    auto& slot = memory_graphs_[cache_.cap];
    if (slot != nullptr) return *slot;
    slot = std::make_unique<MemoryGraph>();
    MemoryGraph& mg = *slot;
    mg.cap = cache_.cap;
    make_weighted(w, &mg.g);
    zf::apple::Graph& g = mg.g.g;
    Builder net(g, mg.g.params, w.layers.size());
    mg.m = net.rows(mg.in, mg.cap);
    Node loss = g.zeros({1});
    for (const auto& kv : net.memory_side(mg.m.rows, mg.m.mask)) {
      mg.k.push_back(kv.k);
      mg.v.push_back(kv.v);
      mg.dk.push_back(g.input({mg.cap, kHD}));
      mg.dv.push_back(g.input({mg.cap, kHD}));
      loss = g.add(loss, g.add(dot_sum(g, kv.k, mg.dk.back()), dot_sum(g, kv.v, mg.dv.back())));
      mg.k_out.emplace_back(static_cast<std::size_t>(mg.cap) * HD);
      mg.v_out.emplace_back(static_cast<std::size_t>(mg.cap) * HD);
    }
    mg.d_rows = g.input({mg.cap, kS});
    loss = g.add(loss, dot_sum(g, mg.m.rows, mg.d_rows));
    weigh(mg.g, loss, {});
    mg.rows_out = Buffer(g.elements(mg.m.rows));
    return mg;
  }

  PassGraph& pass_graph(Weights& w, std::size_t n) {
    cap_b_ = capacity(n, 8);
    if (cache_.cap == 0) cache_.grow(64, 0, w.layers.size());
    auto& slot = pass_graphs_[{cap_b_, cache_.cap}];
    if (slot != nullptr) return *slot;
    slot = std::make_unique<PassGraph>();
    PassGraph& pg = *slot;
    pg.batch = cap_b_;
    make_weighted(w, &pg.g);
    zf::apple::Graph& g = pg.g.g;
    Builder net(g, pg.g.params, w.layers.size());
    pg.q = net.queries(pg.in, pg.batch, kQ);
    pg.cbias = pg.in.add(g, {pg.batch, 1, 1, kM});
    pg.idx = pg.in.add(g, {pg.batch * kM}, true);
    pg.d_hidden = pg.in.add(g, {pg.batch * kQ, kH});
    std::vector<typename Builder::KV> kv;
    for (std::size_t l = 0; l < w.layers.size(); ++l) {
      pg.k.push_back(g.input({cache_.cap, kHD}));
      pg.v.push_back(g.input({cache_.cap, kHD}));
      kv.push_back({g.gather(pg.k.back(), pg.idx, 0), g.gather(pg.v.back(), pg.idx, 0)});
    }
    pg.hidden = net.hidden(net.stack(pg.q, kv, pg.cbias, kM));
    std::vector<Node> extra{pg.q.rows.x};
    extra.insert(extra.end(), pg.k.begin(), pg.k.end());
    extra.insert(extra.end(), pg.v.begin(), pg.v.end());
    const std::vector<Node> d = weigh(pg.g, dot_sum(g, pg.hidden, pg.d_hidden), extra);
    pg.dx = d[0];
    const std::size_t levels = w.layers.size();
    for (std::size_t l = 0; l < levels; ++l) {
      pg.gk_in.push_back(g.input({cache_.cap, kHD}));
      pg.gv_in.push_back(g.input({cache_.cap, kHD}));
      pg.gk_out.push_back(g.add(pg.gk_in.back(), d[1 + l]));
      pg.gv_out.push_back(g.add(pg.gv_in.back(), d[1 + levels + l]));
    }
    pg.hidden_out = Buffer(g.elements(pg.hidden));
    pg.dx_out = Buffer(g.elements(pg.dx));
    return pg;
  }

  // The memories appended to the cache: their rows through the memory
  // graph together, K and V copied in after the rows already there.
  void add_memories(Weights& w, const std::vector<const std::vector<Input>*>& memories) {
    if (pending_) throw std::logic_error("LowResCross: memories changed before backward_memories()");
    const std::size_t r0 = memory_.size();
    for (const auto* rows : memories) {
      sb_.push_back(memory_.size());
      sl_.push_back(rows->size());
      memory_.insert(memory_.end(), rows->begin(), rows->end());
    }
    const int need = capacity(memory_.size(), 64);
    if (need > cache_.cap) cache_.grow(need, r0, w.layers.size());
    // The rows already there were made from older weights: all again.
    encode_memories(w, kv_version_ == w.version ? r0 : 0);
  }

  // The cache's rows from r0 on through the memory side: their rows, K and
  // V into the cache.
  void encode_memories(Weights& w, std::size_t r0) {
    if (pending_) throw std::logic_error("LowResCross: weights changed before backward_memories()");
    kv_version_ = w.version;
    const std::size_t n = memory_.size() - r0;
    if (n == 0) return;
    MemoryGraph& mg = memory_graph(w);
    upload(w);
    mg.in.zero();
    Builder::clear_ids(mg.in, mg.m);
    for (std::size_t r = 0; r < n; ++r) Builder::put_row(mg.in, mg.m, static_cast<int>(r), memory_[r0 + r]);
    std::vector<Bind> in, out{{mg.m.rows, mg.rows_out}};
    bind_inputs(mg.g, mg.in, &in);
    for (std::size_t l = 0; l < mg.k.size(); ++l) {
      out.push_back({mg.k[l], mg.k_out[l]});
      out.push_back({mg.v[l], mg.v_out[l]});
    }
    mg.g.g.run(in, out);
    const auto put = [&](const Buffer& from, Buffer& to, std::size_t width) {
      std::memcpy(to.data() + r0 * width, from.data(), n * width * sizeof(float));
    };
    put(mg.rows_out, cache_.rows, kStream);
    for (std::size_t l = 0; l < mg.k.size(); ++l) {
      put(mg.k_out[l], cache_.k[l], HD);
      put(mg.v_out[l], cache_.v[l], HD);
    }
  }

  // The facade's weights into the buffers every graph reads, when they
  // have changed since.
  void upload(Weights& w) {
    if (uploaded_ == w.version) return;
    uploaded_ = w.version;
    const bool fresh = weights_.empty();
    std::size_t i = 0;
    zf::visit_params(w, [&](auto& t) {
      if (fresh) {
        weights_.emplace_back(t.size());
        sums_.emplace_back(t.size());
        sums_next_.emplace_back(t.size());
      }
      std::memcpy(weights_.at(i++).data(), t.data(), t.size() * sizeof(float));
    });
  }
  void bind_inputs(const Weighted& wg, const LowResGraphInputs& gi, std::vector<Bind>* in) const {
    const std::vector<Node>& nodes = wg.params.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) in->push_back({nodes[i], weights_.at(i)});
    gi.bind(in);
  }
  // A run that computes the weights' gradients: the first of a sequence
  // writes them into the sum, later ones add to it.
  void run_with_grads(Weighted& wg, std::vector<Bind>* in, std::vector<Bind>* out) {
    for (std::size_t i = 0; i < wg.grads.size(); ++i) {
      if (summing_) {
        in->push_back({wg.sum_in[i], sums_.at(i)});
        out->push_back({wg.sum_out[i], sums_next_.at(i)});
      } else {
        out->push_back({wg.grads[i], sums_.at(i)});
      }
    }
    wg.g.run(*in, *out);
    if (summing_) sums_.swap(sums_next_);
    summing_ = true;
  }
  // The sequence's gradients into the facade's.
  void flush_grads(Weights& w) {
    if (summing_) zf::apple::add_grads(sums_, w);
    summing_ = false;
  }

  // `batch` into a block's inputs (every input zeroed first), and the
  // batch's bookkeeping.
  void pack(Block& blk, const std::vector<Pass>& batch) {
    blk.in.zero();
    Builder::clear_queries(blk.in, blk.q);
    Builder::clear_ids(blk.in, blk.m);
    float* cbias = blk.in[blk.cbias].data();
    std::fill(cbias, cbias + blk.batch * kM, Builder::kNoKey);
    begin_batch(batch.size());
    for (std::size_t b = 0; b < batch.size(); ++b) {
      const std::vector<Input>& ms = *batch[b].memory;
      if (ms.size() > M) throw std::length_error("LowResCross overflow");
      Builder::put_queries(blk.in, blk.q, static_cast<int>(b), *batch[b].queries, !ms.empty());
      for (std::size_t j = 0; j < ms.size(); ++j) {
        const int row = static_cast<int>(b) * kM + static_cast<int>(j);
        Builder::put_row(blk.in, blk.m, row, ms[j]);
        cbias[row] = 0.0f;
      }
      add_pass(*batch[b].queries, b * M, ms.size());
    }
  }

  void begin_batch(std::size_t n) {
    qo_.assign(1, 0);
    mb_.clear();
    ml_.clear();
    queries_.clear();
    queries_.reserve(n * Q);
  }
  void add_pass(const std::vector<Input>& queries, std::size_t mb, std::size_t ml) {
    queries_.insert(queries_.end(), queries.begin(), queries.end());
    qo_.push_back(queries_.size());
    mb_.push_back(mb);
    ml_.push_back(ml);
  }
  void fill_d_hidden(const std::vector<zf::Tensor<T, Q, kHidden>>& d_readout, Buffer* d_hidden) const {
    d_hidden->zero();
    for (std::size_t b = 0; b < d_readout.size(); ++b) {
      for (std::size_t i = 0; i < query_size(b); ++i) {
        float* d = d_hidden->data() + (b * Q + i) * kHidden;
        for (std::size_t k = 0; k < kHidden; ++k) d[k] = d_readout[b](i, k);
      }
    }
  }

  Mode mode_ = Mode::kBatch;
  int cap_b_ = 0;
  std::map<int, std::unique_ptr<BatchGraph>> batch_graphs_;
  std::map<int, std::unique_ptr<MemoryGraph>> memory_graphs_;
  std::map<std::pair<int, int>, std::unique_ptr<PassGraph>> pass_graphs_;
  std::map<std::pair<std::type_index, int>, std::unique_ptr<AnyStep>> step_graphs_;
  // The weights as of version uploaded_; the gradients' running sum
  // (summing_: a run has written it since the last flush).
  std::vector<Buffer> weights_, sums_, sums_next_;
  std::uint64_t uploaded_ = 0;
  bool summing_ = false;

  // The cache: memory j's rows are sl_[j] from sb_[j], in memory_ and in
  // the cache's rows; pending_ while their gradients wait for
  // backward_memories().
  Cache cache_;
  std::vector<Input> memory_;
  std::vector<std::size_t> sb_, sl_;
  bool pending_ = false;
  // The weights' version K and V were made from.
  std::uint64_t kv_version_ = 0;

  // The batch just run: pass b's queries are [qo_[b], qo_[b+1]) of
  // queries_, its memory rows ml_[b] from mb_[b] of rows_.
  std::vector<std::size_t> qo_{0}, mb_, ml_;
  std::vector<Input> queries_;
  Buffer hidden_, rows_, d_rows_;
};

}  // namespace zf

#endif  // ZTENSOR_APPLE_GPU
