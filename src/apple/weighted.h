#pragma once

// Graphs that read a CPU model's weights and hand their gradients back, for
// a model's GPU implementation (lowrescross_apple.h, focuscross_apple.h).
//
// The CPU model stays the master copy: the runs read a copy of its weights,
// taken again only when their version has moved (once a step), and the
// gradients come back into the model's, so the step stays on the CPU. The
// gradients add up on the GPU over a sequence of runs (each run adds its own
// into a running sum) and come back once, at the sequence's end.

#if defined(ZTENSOR_APPLE_GPU)

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "apple/graph.h"
#include "apple/params.h"

namespace zf {
namespace apple {

// A graph with the weights as its inputs and their gradients: alone, and
// added to a running sum (sum_in + grad).
struct Weighted {
  Graph g;
  Params params;
  std::vector<Node> grads, sum_in, sum_out;

  template <typename Model>
  void declare(Model& model) {
    params.declare(g, model);
  }
  // The gradients of `loss` on the weights (and on `extra`, returned).
  std::vector<Node> weigh(Node loss, const std::vector<Node>& extra) {
    std::vector<Node> wrt = params.nodes();
    wrt.insert(wrt.end(), extra.begin(), extra.end());
    std::vector<Node> all = g.gradients(loss, wrt);
    grads.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(params.size()));
    for (const Node grad : grads) {
      sum_in.push_back(g.input(g.shape(grad)));
      sum_out.push_back(g.add(sum_in.back(), grad));
    }
    return std::vector<Node>(all.begin() + static_cast<std::ptrdiff_t>(params.size()), all.end());
  }
};

// The weights every graph of a model reads, and their gradients' running
// sum.
class WeightRuns {
 public:
  // The model's weights into the buffers, when they have changed since
  // (`version`, which the model moves on every change).
  template <typename Model>
  void upload(Model& w) {
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
  // The weights as a run's inputs.
  void bind(const Weighted& wg, std::vector<Graph::Bind>* in) const {
    const std::vector<Node>& nodes = wg.params.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) in->push_back({nodes[i], weights_.at(i)});
  }
  // A run that computes the weights' gradients: the first of a sequence
  // writes them into the sum, later ones add to it.
  void run_with_grads(Weighted& wg, std::vector<Graph::Bind>* in, std::vector<Graph::Bind>* out) {
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
  // The sequence's gradients into the model's.
  template <typename Model>
  void flush(Model& w) {
    if (summing_) add_grads(sums_, w);
    summing_ = false;
  }

 private:
  // The weights as of version uploaded_; the gradients' running sum
  // (summing_: a run has written it since the last flush).
  std::vector<Buffer> weights_, sums_, sums_next_;
  std::uint64_t uploaded_ = 0;
  bool summing_ = false;
};

}  // namespace apple
}  // namespace zf

#endif  // ZTENSOR_APPLE_GPU
