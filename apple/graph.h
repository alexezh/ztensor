#pragma once

// zf::apple: training graphs on the GPU, on Metal Performance Shaders Graph.
//
// The CPU library runs a model one layer at a time, with hand-written
// backward passes. Here a whole training step -- forward, loss, gradients,
// the update -- is one graph that runs on the GPU in one go. It is built
// once from the ops below, gets its gradients from MPSGraph's autodiff
// (gradients()), and is compiled on its first run. A GPU step has a fixed
// cost of a few milliseconds, so this pays only for big batches (a few
// hundred examples).
//
// Shapes are fixed when the graph is built: a batch is padded to them, and
// masks keep the padding out. Everything is float32, except integer ids
// (input_ints, one_hot).
//
// Data lives in Buffers: memory the CPU and the GPU share (Apple silicon
// has one memory), so feeding an input or reading a result is a plain
// store or load, no copy. A run binds buffers to the graph's inputs and
// outputs. A buffer that is one run's output and a later run's input keeps
// state on the GPU between runs: the weights a step updates (params.h), a
// cache of keys and values.
//
// This header is plain C++; the Objective-C++ is in graph.mm. nn.h builds
// layers, attention, losses and the SGD step from these ops; params.h puts
// a CPU model's parameters on the GPU and back.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace zf {
namespace apple {

// Whether this machine has a Metal device.
bool available();

// Memory the CPU and the GPU share: n floats or n int32s. Copies share the
// memory (a handle).
class Buffer {
 public:
  enum class Type { kFloat, kInt };

  Buffer() = default;
  explicit Buffer(std::size_t n, Type type = Type::kFloat);

  bool empty() const { return impl_ == nullptr; }
  std::size_t size() const;
  Type type() const;
  float* data();
  const float* data() const;
  std::int32_t* ints();
  const std::int32_t* ints() const;
  void zero();

 private:
  friend class Graph;
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// A tensor of the graph being built.
struct Node {
  int id = -1;
  bool valid() const { return id >= 0; }
};

using Shape = std::vector<int>;

class Graph {
 public:
  Graph();
  ~Graph();
  Graph(Graph&&) noexcept;
  Graph& operator=(Graph&&) noexcept;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  // Fed by a buffer at every run.
  Node input(const Shape& shape);
  Node input_ints(const Shape& shape);

  Shape shape(Node n) const;
  std::size_t elements(Node n) const;

  // A scalar; broadcasts like any operand.
  Node constant(double v);
  Node zeros(const Shape& shape);

  // Elementwise, with numpy-style broadcasting.
  Node add(Node a, Node b);
  Node sub(Node a, Node b);
  Node mul(Node a, Node b);
  Node div(Node a, Node b);
  Node maximum(Node a, Node b);
  Node minimum(Node a, Node b);
  Node neg(Node a);
  Node exp(Node a);
  Node log(Node a);
  Node abs(Node a);
  Node sqrt(Node a);
  Node rsqrt(Node a);
  Node square(Node a);
  Node relu(Node a);
  Node sigmoid(Node a);
  // 1 where a > b, else 0.
  Node greater(Node a, Node b);

  // Reductions over `axes` (negative counts from the end); the reduced axes
  // stay, with length 1. MPSGraph has no gradient for max: a loss must not
  // depend on it.
  Node sum(Node a, const std::vector<int>& axes);
  Node max(Node a, const std::vector<int>& axes);
  // Every element, to a scalar of shape [1].
  Node sum_all(Node a);

  // [..., n, k] x [..., k, m] -> [..., n, m]; leading axes broadcast.
  Node matmul(Node a, Node b);
  Node transpose(Node a, int axis0, int axis1);
  Node reshape(Node a, const Shape& shape);
  Node slice(Node a, int axis, int start, int length);
  // Zeros before and after along `axis`.
  Node pad(Node a, int axis, int before, int after);
  Node concat(const std::vector<Node>& parts, int axis);
  // ids (int32) -> one-hot rows of `depth` floats on a new last axis. An id
  // outside [0, depth) gives a row of zeros.
  Node one_hot(Node ids, int depth);
  // The slices of `a` along `axis` at `ids` (int32, each in range): a [R, d]
  // gathered at ids [n] is [n, d]. Its gradient adds into the rows gathered.
  Node gather(Node a, Node ids, int axis);
  Node softmax(Node a, int axis);
  // zf::softmax_ce per row: -sum(target log softmax(logits)) over the last
  // axis, with a one-hot (or any distribution) target; [..., 1]. Stable for
  // any logits. A class that must not count gets a large negative bias on
  // its logit, as zf::softmax_ce's `n` leaves it out.
  Node softmax_ce(Node logits, Node target);

  // d(loss)/d(each of wrt). `loss` must be a scalar ([1] or []). A tensor
  // the loss does not depend on gets zeros.
  std::vector<Node> gradients(Node loss, const std::vector<Node>& wrt);

  // One run: every input bound to a buffer of its size, the outputs written
  // into theirs. Returns when the GPU is done.
  struct Bind {
    Node node;
    Buffer buffer;
  };
  void run(const std::vector<Bind>& inputs, const std::vector<Bind>& outputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace apple
}  // namespace zf
