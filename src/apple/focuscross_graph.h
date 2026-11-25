#pragma once

// FocusCross as a zf::apple graph, for the GPU implementation
// (focuscross_apple.h): LowResCross's pieces (lowrescross_graph.h) -- the
// weights as graph inputs, blocks of rows, packing -- and a stack whose
// workspace rows attend to each other at every level, gated by the relevance
// the level before left (focuscross.h has the maths).
//
// Padding is masked as in LowResCross: a padded workspace row is no key in
// either attention and writes nothing; a padded query row adds nothing to
// the relevance.

#if defined(ZTENSOR_APPLE_GPU)

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "apple/graph.h"
#include "apple/lowrescross_graph.h"
#include "apple/nn.h"
#include "apple/params.h"
#include "focuscross_weights.h"

namespace zf {

template <typename T, typename Layout, std::size_t Heads, std::size_t HD>
class FocusCrossGraph : public LowResCrossGraph<T, Layout, Heads, HD> {
  using Base = LowResCrossGraph<T, Layout, Heads, HD>;
  using typename Base::Lin;
  using typename Base::Norm;
  using Base::g_;

 public:
  using Node = zf::apple::Node;
  using typename Base::Queries;
  using typename Base::Rows;

  // What the stack leaves: the query rows and the workspace rows after the
  // last level, and each level's relevance, [batch * m].
  struct Result {
    Node final_rows, rows;
    std::vector<Node> relevance;
  };

  // The weights, taken from `params` in FocusCrossWeights' visit order.
  FocusCrossGraph(zf::apple::Graph& g, zf::apple::Params& params, std::size_t levels)
      : Base(g, params, levels, false) {
    for (std::size_t l = 0; l < levels; ++l) {
      Focus f;
      f.nw = Norm{params.next(), params.next()};
      for (Lin* p : {&f.qw, &f.kw, &f.vw, &f.ow}) *p = Lin{params.next(), params.next()};
      focus_.push_back(f);
    }
    if (!params.all_taken()) throw std::logic_error("FocusCrossGraph: parameters left over");
  }

  // The stack: `qs` over the workspace `ws` ([batch * m] rows; `wmask`
  // [batch * m, kHidden] where they may write; `cbias` [batch, 1, 1, m]
  // which are rows).
  Result stack(const Queries& qs, const Rows& ws, Node wmask, Node cbias, int m) {
    zf::apple::Graph& g = g_;
    const int b = qs.batch, q = qs.q;
    const auto qln = [&](const Norm& p, Node in) {
      return zf::apple::masked_layer_norm(g, in, qs.rows.mask, p.gamma, p.beta);
    };
    const auto wln = [&](const Norm& p, Node in) {
      return zf::apple::masked_layer_norm(g, in, ws.mask, p.gamma, p.beta);
    };
    // The real query rows, [b, 1, q, 1]: sbias is 0 for a row, kNoKey for padding.
    const Node live = g.reshape(g.greater(qs.sbias, g.constant(-1.0)), {b, 1, q, 1});
    Result out;
    Node x = qs.rows.rows, w = ws.rows;
    Node r;  // [b, 1, 1, m]; none before the first cross attention
    for (std::size_t l = 0; l < this->layers_.size(); ++l) {
      const auto& wl = this->layers_[l];
      const Focus& wf = focus_[l];
      // workspace: w_j += r_j · mask · O_w(attn(..., keys biased by log(r + eps)))
      // (at the first level r = 1: no gate, an even bias that changes nothing)
      Node wn = wln(wf.nw, w);
      const Node bias = r.valid() ? g.add(cbias, g.log(g.add(r, g.constant(kFocusEps)))) : cbias;
      Node uw = this->apply(wf.ow, attend(this->apply(wf.qw, wn), this->apply(wf.kw, wn), this->apply(wf.vw, wn),
                                          bias, b, m, m).first);
      if (r.valid()) uw = g.mul(uw, g.reshape(r, {b * m, 1}));
      w = write(w, uw, wmask);
      // query self attention
      Node xn = qln(wl.n1, x);
      x = write(x,
                this->apply(wl.os, attend(this->apply(wl.qs, xn), this->apply(wl.ks, xn), this->apply(wl.vs, xn),
                                          qs.sbias, b, q, q).first),
                qs.wmask);
      // cross attention, and the relevance it reads: r = 1 - exp(-s), s the
      // attention a row got, summed over the real query rows, over heads
      xn = qln(wl.n2, x);
      const Node mn = wln(wl.nm, w);
      const auto [ctx, p] = attend(this->apply(wl.qc, xn), this->apply(wl.kc, mn), this->apply(wl.vc, mn), cbias, b,
                                   q, m);
      x = write(x, this->apply(wl.oc, ctx), qs.cmask);
      const Node s = g.div(g.sum(g.mul(p, live), {1, 2}), g.constant(static_cast<double>(Heads)));
      r = g.sub(g.constant(1.0), g.exp(g.neg(s)));
      out.relevance.push_back(g.reshape(r, {b * m}));
      // feed-forward
      xn = qln(wl.n3, x);
      x = write(x, this->apply(wl.f2, g.relu(this->apply(wl.f1, xn))), qs.wmask);
    }
    out.final_rows = x;
    out.rows = w;
    return out;
  }

 private:
  struct Focus {
    Norm nw;
    Lin qw, kw, vw, ow;
  };

  // x (rows of kStream) += mask · u (rows of kHidden), into the state.
  Node write(Node x0, Node u, Node mask) {
    const int at = static_cast<int>(Base::kHiddenOffset);
    return g_.add(x0, g_.pad(g_.mul(u, mask), 1, at, Base::kS - at - Base::kH));
  }

  // nn.h's attention, with its probabilities too: [batch, heads, nq, nk].
  std::pair<Node, Node> attend(Node q, Node k, Node v, Node key_bias, int batch, int nq, int nk) {
    zf::apple::Graph& g = g_;
    const int heads = static_cast<int>(Heads);
    const int dim = static_cast<int>(HD) / heads;
    const auto split = [&](Node x, int n) {
      return g.transpose(g.reshape(x, {batch, n, heads, dim}), 1, 2);  // [batch, heads, n, dim]
    };
    Node s = g.matmul(split(q, nq), g.transpose(split(k, nk), 2, 3));
    s = g.add(g.mul(s, g.constant(1.0 / std::sqrt(static_cast<double>(dim)))), key_bias);
    const Node p = g.softmax(s, 3);
    const Node ctx = g.matmul(p, split(v, nk));
    return {g.reshape(g.transpose(ctx, 1, 2), {batch * nq, static_cast<int>(HD)}), p};
  }

  std::vector<Focus> focus_;
};

}  // namespace zf

#endif  // ZTENSOR_APPLE_GPU
