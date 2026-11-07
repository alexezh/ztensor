#pragma once

// The pieces of LowResCross as a zf::apple graph, for the GPU
// implementation (lowrescross_apple.h) to put together: the weights as
// graph inputs, blocks of rows, the memory side (each layer's cross K and
// V) and the query stack.
//
// Shapes are fixed when a graph is built, so rows come in padded blocks.
// The rows are what the CPU builds (Layout::put, Layout::put_mask); only
// the learned id fields are added on the GPU, from one-hot ids, so their
// gradients reach the embeddings. Padding is masked out: a padded query row
// writes nothing and is no key, a padded memory row is no key.

#if defined(ZTENSOR_APPLE_GPU)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "apple/graph.h"
#include "apple/nn.h"
#include "apple/params.h"
#include "lowrescross_layout.h"

namespace zf {

// Graph inputs, each with the buffer that feeds it.
class LowResGraphInputs {
 public:
  zf::apple::Node add(zf::apple::Graph& g, const zf::apple::Shape& shape, bool ints = false) {
    const zf::apple::Node n = ints ? g.input_ints(shape) : g.input(shape);
    binds_.push_back({n, zf::apple::Buffer(g.elements(n), ints ? zf::apple::Buffer::Type::kInt
                                                                : zf::apple::Buffer::Type::kFloat)});
    return n;
  }
  zf::apple::Buffer& operator[](zf::apple::Node n) {
    for (auto& b : binds_) {
      if (b.node.id == n.id) return b.buffer;
    }
    throw std::logic_error("LowResGraphInputs: no such input");
  }
  void zero() {
    for (auto& b : binds_) b.buffer.zero();
  }
  void bind(std::vector<zf::apple::Graph::Bind>* into) const { into->insert(into->end(), binds_.begin(), binds_.end()); }

 private:
  std::vector<zf::apple::Graph::Bind> binds_;
};

template <typename T, typename Layout, std::size_t Heads, std::size_t HD>
class LowResCrossGraph {
 public:
  using Node = zf::apple::Node;
  using Input = typename Layout::Input;
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  static constexpr std::size_t kHiddenOffset = Layout::kStateOffset;
  static constexpr int kS = static_cast<int>(kStream);
  static constexpr int kH = static_cast<int>(kHidden);
  static constexpr int kSlots = static_cast<int>(Layout::kIdSlots);
  static constexpr int kHD = static_cast<int>(HD);
  // A key that is padding: its score before the softmax.
  static constexpr float kNoKey = -1e9f;

  // A block of n rows: their fixed part, read mask and ids, and the rows as
  // the stack reads them (with the id fields).
  struct Rows {
    int n = 0;
    Node x, mask, ids;
    Node rows;
  };
  // The query rows of `batch` passes of q rows each, with where the layers
  // write and which rows are keys.
  struct Queries {
    int batch = 0, q = 0;
    Rows rows;
    Node wmask, cmask, sbias;
  };
  struct KV {
    Node k, v;
  };

  // The weights, taken from `params` in LowResCrossWeights' visit order.
  LowResCrossGraph(zf::apple::Graph& g, zf::apple::Params& params, std::size_t levels) : g_(g) {
    for (int s = 0; s < kSlots; ++s) ids_.push_back(params.next());
    for (std::size_t l = 0; l < levels; ++l) {
      Layer w;
      for (Norm* n : {&w.n1, &w.n2, &w.nm, &w.n3}) *n = Norm{params.next(), params.next()};
      for (Lin* p : {&w.qs, &w.ks, &w.vs, &w.qc, &w.kc, &w.vc, &w.os, &w.oc, &w.f1, &w.f2}) {
        *p = Lin{params.next(), params.next()};
      }
      layers_.push_back(w);
    }
    if (!params.all_taken()) throw std::logic_error("LowResCrossGraph: parameters left over");
  }

  Rows rows(LowResGraphInputs& in, int n) {
    Rows r;
    r.n = n;
    r.x = in.add(g_, {n, kS});
    r.mask = in.add(g_, {n, kS});
    r.ids = in.add(g_, {n, kSlots}, true);
    // The id fields: each slot's one-hot id times its embeddings, added at
    // its place in the row.
    const int tables = static_cast<int>(Layout::kIds);
    const int dim = static_cast<int>(Layout::kIdDim);
    const Node hot = g_.one_hot(r.ids, tables);  // [n, kSlots, kIds]
    r.rows = r.x;
    for (int s = 0; s < kSlots; ++s) {
      const Node e = g_.matmul(g_.reshape(g_.slice(hot, 1, s, 1), {n, tables}), ids_[s]);
      const int at = static_cast<int>(Layout::kIdAt[s]);
      r.rows = g_.add(r.rows, g_.pad(e, 1, at, kS - at - dim));
    }
    return r;
  }

  Queries queries(LowResGraphInputs& in, int batch, int q) {
    Queries out;
    out.batch = batch;
    out.q = q;
    out.rows = rows(in, batch * q);
    out.wmask = in.add(g_, {batch * q, kH});
    out.cmask = in.add(g_, {batch * q, kH});
    out.sbias = in.add(g_, {batch, 1, 1, q});
    return out;
  }

  // Each layer's cross K and V of memory rows [n, kStream] (read mask
  // `mask`): [n, HD] each.
  std::vector<KV> memory_side(Node rows, Node mask) {
    std::vector<KV> out;
    for (const Layer& w : layers_) {
      const Node mn = zf::apple::masked_layer_norm(g_, rows, mask, w.nm.gamma, w.nm.beta);
      out.push_back({apply(w.kc, mn), apply(w.vc, mn)});
    }
    return out;
  }

  // The stack over `kv` (per layer, [batch * m, HD], m keys per pass;
  // cbias [batch, 1, 1, m]). Returns the final rows [batch * q, kStream].
  Node stack(const Queries& qs, const std::vector<KV>& kv, Node cbias, int m) {
    const int heads = static_cast<int>(Heads);
    const auto write = [&](Node x0, Node u, Node mask) {
      const int at = static_cast<int>(kHiddenOffset);
      return g_.add(x0, g_.pad(g_.mul(u, mask), 1, at, kS - at - kH));
    };
    const auto ln = [&](const Norm& p, Node in) {
      return zf::apple::masked_layer_norm(g_, in, qs.rows.mask, p.gamma, p.beta);
    };
    Node x = qs.rows.rows;
    for (std::size_t l = 0; l < layers_.size(); ++l) {
      const Layer& w = layers_[l];
      // self
      Node xn = ln(w.n1, x);
      Node ctx = zf::apple::attention(g_, apply(w.qs, xn), apply(w.ks, xn), apply(w.vs, xn), qs.sbias, qs.batch,
                                      qs.q, qs.q, heads);
      x = write(x, apply(w.os, ctx), qs.wmask);
      // cross
      xn = ln(w.n2, x);
      ctx = zf::apple::attention(g_, apply(w.qc, xn), kv[l].k, kv[l].v, cbias, qs.batch, qs.q, m, heads);
      x = write(x, apply(w.oc, ctx), qs.cmask);
      // feed-forward
      xn = ln(w.n3, x);
      x = write(x, apply(w.f2, g_.relu(apply(w.f1, xn))), qs.wmask);
    }
    return x;
  }
  Node hidden(Node final_rows) { return g_.slice(final_rows, 1, static_cast<int>(kHiddenOffset), kH); }

  // --- packing: over buffers zeroed first ---

  // Row i of a block.
  static void put_row(LowResGraphInputs& in, const Rows& r, int i, const Input& t) {
    Layout::put(in[r.x].data() + static_cast<std::size_t>(i) * kStream, t);
    Layout::put_mask(in[r.mask].data() + static_cast<std::size_t>(i) * kStream, t);
    // Id -1 is none: an id the one-hot leaves at zero.
    std::int32_t* ids = in[r.ids].ints() + static_cast<std::size_t>(i) * kSlots;
    for (int s = 0; s < kSlots; ++s) ids[s] = Layout::id(t, static_cast<std::size_t>(s));
  }
  // A block's ids where it has no row: none.
  static void clear_ids(LowResGraphInputs& in, const Rows& r) {
    std::fill_n(in[r.ids].ints(), static_cast<std::size_t>(r.n) * kSlots, -1);
  }
  // Pass b's query rows; `has_memory`: whether its cross attention writes.
  static void put_queries(LowResGraphInputs& in, const Queries& qs, int b, const std::vector<Input>& rows,
                          bool has_memory) {
    if (static_cast<int>(rows.size()) > qs.q) throw std::length_error("LowResCrossGraph: too many queries");
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const int row = b * qs.q + static_cast<int>(i);
      put_row(in, qs.rows, row, rows[i]);
      in[qs.sbias].data()[row] = 0.0f;
      Layout::put_write_mask(in[qs.wmask].data() + static_cast<std::size_t>(row) * kHidden, rows[i]);
      if (has_memory) Layout::put_write_mask(in[qs.cmask].data() + static_cast<std::size_t>(row) * kHidden, rows[i]);
    }
  }
  // Before any put_queries: every query row padding.
  static void clear_queries(LowResGraphInputs& in, const Queries& qs) {
    clear_ids(in, qs.rows);
    float* s = in[qs.sbias].data();
    std::fill(s, s + qs.batch * qs.q, kNoKey);
  }

 private:
  struct Norm {
    Node gamma, beta;
  };
  struct Lin {
    Node w, b;
  };
  struct Layer {
    Norm n1, n2, nm, n3;
    Lin qs, ks, vs, qc, kc, vc, os, oc, f1, f2;
  };

  Node apply(const Lin& p, Node in) { return zf::apple::linear(g_, in, p.w, p.b); }

  zf::apple::Graph& g_;
  std::vector<Node> ids_;
  std::vector<Layer> layers_;
};

}  // namespace zf

#endif  // ZTENSOR_APPLE_GPU
