#pragma once

// Layers, attention, losses and the SGD step, built from Graph's ops. Each
// computes what its CPU counterpart in ztensor computes, so a model can
// run on either: weights are in the CPU layout (a Linear's weight is
// [out, in]), the masked layer norm is zf::LayerNorm's masked path, the
// losses are zf::softmax_ce and zf::bce, and clipped_sgd is the clipped
// step of a batch.

#include <cmath>
#include <vector>

#include "graph.h"

namespace zf {
namespace apple {

// x [rows, in] -> x W^T + b, with W [out, in] and b [out] as zf::Linear holds
// them.
inline Node linear(Graph& g, Node x, Node w, Node b) {
  return g.add(g.matmul(x, g.transpose(w, 0, 1)), b);
}

// zf::LayerNorm's masked path over the last axis: statistics over the slots
// whose mask is 1, and the masked-out slots zero on the output.
//   out = m (gamma (x - mu) / sqrt(var + eps) + beta)
// A row with no slots at all comes out zero.
inline Node masked_layer_norm(Graph& g, Node x, Node mask, Node gamma, Node beta, double eps = 1e-5) {
  const Node n = g.maximum(g.sum(mask, {-1}), g.constant(1.0));
  const Node mean = g.div(g.sum(g.mul(x, mask), {-1}), n);
  const Node xc = g.sub(x, mean);
  const Node var = g.div(g.sum(g.mul(mask, g.square(xc)), {-1}), n);
  const Node xhat = g.mul(xc, g.rsqrt(g.add(var, g.constant(eps))));
  return g.mul(mask, g.add(g.mul(xhat, gamma), beta));
}

// Scaled dot-product attention, `batch` passes at once: q is the passes'
// query rows stacked [batch * nq, heads * dim], k and v their key rows
// [batch * nk, heads * dim]. key_bias [batch, 1, 1, nk] is added to the
// scores before the softmax: 0 for a key, a large negative number for
// padding. Returns the context rows, [batch * nq, heads * dim].
inline Node attention(Graph& g, Node q, Node k, Node v, Node key_bias, int batch, int nq, int nk, int heads) {
  const int hd = g.shape(q).back();
  const int dim = hd / heads;
  const auto split = [&](Node x, int n) {
    return g.transpose(g.reshape(x, {batch, n, heads, dim}), 1, 2);  // [batch, heads, n, dim]
  };
  const Node qh = split(q, nq), kh = split(k, nk), vh = split(v, nk);
  Node s = g.matmul(qh, g.transpose(kh, 2, 3));
  s = g.add(g.mul(s, g.constant(1.0 / std::sqrt(static_cast<double>(dim)))), key_bias);
  const Node ctx = g.matmul(g.softmax(s, 3), vh);  // [batch, heads, nq, dim]
  return g.reshape(g.transpose(ctx, 1, 2), {batch * nq, hd});
}

// zf::bce on logits z and 0/1 targets y, elementwise and stable:
// max(z, 0) - z y + log(1 + exp(-|z|)).
inline Node bce_logits(Graph& g, Node z, Node y) {
  return g.add(g.sub(g.relu(z), g.mul(z, y)), g.log(g.add(g.constant(1.0), g.exp(g.neg(g.abs(z))))));
}

// One SGD step on `params` with their `grads`, held to a global gradient
// norm of max_norm: p - lr min(1, max_norm / |grads|) g. lr and max_norm
// are inputs of shape [1], so a run can change them. Returns the updated
// parameters, in order.
inline std::vector<Node> clipped_sgd(Graph& g, const std::vector<Node>& params, const std::vector<Node>& grads,
                                     Node lr, Node max_norm) {
  Node norm2 = g.constant(0.0);
  for (Node d : grads) norm2 = g.add(norm2, g.sum_all(g.square(d)));
  const Node norm = g.maximum(g.sqrt(norm2), g.constant(1e-30));
  const Node step = g.mul(lr, g.minimum(g.constant(1.0), g.div(max_norm, norm)));
  std::vector<Node> out;
  out.reserve(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) out.push_back(g.sub(params[i], g.mul(step, grads[i])));
  return out;
}

}  // namespace apple
}  // namespace zf
