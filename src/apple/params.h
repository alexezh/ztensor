#pragma once

// A CPU model's parameters on the GPU: one graph input per parameter
// tensor, in zf::visit_params order.
//
// The CPU model stays the master copy (save, load, evaluation). Two ways
// to use the GPU:
//
//   - The CPU steps: upload() the weights before a run, and add_grads() the
//     gradients a run computed into the model's (a graph that only computes
//     gradients).
//   - The GPU steps: upload() once, run steps that read the parameters from
//     one set of buffers and write the updated ones into a second set;
//     flip() then makes the second current, so the weights stay on the GPU
//     from step to step. download() at the end.
//
// A builder takes the inputs with next(), walking the model's layers in
// the same order as visit_params. Several graphs can read the same weights:
// bind() one Params' buffers to another graph's inputs with bind_to().

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "graph.h"
#include "replicas.h"

namespace zf {
namespace apple {

class Params {
 public:
  // The inputs; the buffers come with the first upload().
  template <typename Model>
  void declare(Graph& g, Model& model) {
    nodes_.clear();
    cur_.clear();
    next_.clear();
    cursor_ = 0;
    zf::visit_params(model, [&](auto& t) {
      using Tensor = std::remove_reference_t<decltype(t)>;
      static_assert(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(*t.data())>>, float>,
                    "zf::apple: float parameters only");
      Shape shape;
      for (std::size_t d : Tensor::shape) shape.push_back(static_cast<int>(d));
      nodes_.push_back(g.input(shape));
    });
  }

  // The next declared input.
  Node next() {
    if (cursor_ >= nodes_.size()) throw std::logic_error("zf::apple::Params: more parameters taken than declared");
    return nodes_[cursor_++];
  }
  // Whether the builder took every one.
  bool all_taken() const { return cursor_ == nodes_.size(); }

  std::size_t size() const { return nodes_.size(); }
  const std::vector<Node>& nodes() const { return nodes_; }
  const Buffer& buffer(std::size_t i) const { return cur_.at(i); }

  template <typename Model>
  void upload(Model& model) {
    const bool fresh = cur_.empty();
    std::size_t i = 0;
    zf::visit_params(model, [&](auto& t) {
      if (fresh) cur_.emplace_back(t.size());
      float* to = cur_.at(i++).data();
      for (std::size_t k = 0; k < t.size(); ++k) to[k] = t.data()[k];
    });
  }
  template <typename Model>
  void download(Model& model) {
    std::size_t i = 0;
    zf::visit_params(model, [&](auto& t) {
      const float* from = cur_.at(i++).data();
      for (std::size_t k = 0; k < t.size(); ++k) t.data()[k] = from[k];
    });
  }

  // The parameters as a run's inputs, from the current buffers.
  void bind(std::vector<Graph::Bind>* inputs) const { bind_to(nodes_, inputs); }
  // The current buffers as the inputs `nodes` of another graph, declared
  // from the same model.
  void bind_to(const std::vector<Node>& nodes, std::vector<Graph::Bind>* inputs) const {
    if (nodes.size() != cur_.size()) throw std::logic_error("zf::apple::Params: upload() before a run");
    for (std::size_t i = 0; i < nodes.size(); ++i) inputs->push_back({nodes[i], cur_[i]});
  }
  // `updated` (one per parameter, in order) as a run's outputs, into the
  // other buffers; flip() after the run.
  void bind_updated(const std::vector<Node>& updated, std::vector<Graph::Bind>* outputs) {
    if (updated.size() != nodes_.size()) throw std::invalid_argument("zf::apple::Params: one update per parameter");
    if (next_.empty()) {
      for (const Buffer& b : cur_) next_.emplace_back(b.size());
    }
    for (std::size_t i = 0; i < nodes_.size(); ++i) outputs->push_back({updated[i], next_[i]});
  }
  void flip() { cur_.swap(next_); }

 private:
  std::vector<Node> nodes_;
  std::vector<Buffer> cur_, next_;
  std::size_t cursor_ = 0;
};

// Gradients a run computed (`grads`, one per parameter tensor in
// visit_params order) added into the model's gradients.
template <typename Model>
void add_grads(const std::vector<Buffer>& grads, Model& model) {
  std::size_t i = 0;
  zf::visit_grads(model, [&](auto& t) {
    const float* from = grads.at(i++).data();
    for (std::size_t k = 0; k < t.size(); ++k) t.data()[k] += from[k];
  });
  if (i != grads.size()) throw std::invalid_argument("zf::apple::add_grads: one gradient per parameter");
}

}  // namespace apple
}  // namespace zf
