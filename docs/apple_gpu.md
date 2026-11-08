# Apple GPU

`apple/`: training graphs on the GPU, on Metal Performance Shaders Graph.
Built with `-DZTENSOR_APPLE_GPU=ON`, the default on Apple. The target is
`ztensor_apple`, and it defines `ZTENSOR_APPLE_GPU` for whatever links it.
CPU-only builds don't change.

The CPU library runs a model one layer at a time, with hand-written backward
passes. On the GPU a whole training step (forward, loss, gradients, and
optionally the update) is one graph. It's built once from the ops in
`apple/graph.h`, gets its gradients from MPSGraph's autodiff, and is compiled
on its first run. A GPU run has a fixed cost of a few milliseconds whatever
its size, so the GPU pays only for big batches.

## Graphs

```cpp
zf::apple::Graph g;
const Node x = g.input({rows, in});
const Node y = zf::apple::linear(g, x, w, b);
const Node loss = g.sum_all(g.softmax_ce(y, target));
const auto grads = g.gradients(loss, {w, b});
g.run({{x, xb}, {w, wb}, {b, bb}, {target, tb}}, {{grads[0], dwb}, {grads[1], dbb}});
```

- Shapes are fixed when the graph is built, so a batch is padded to them and
  masks keep the padding out.
- Everything is float32, except integer ids (`input_ints`, `one_hot`,
  `gather`).
- A loss must not depend on `max`, which has no gradient in MPSGraph.
  Comparisons (`greater`) can mask a loss, and they count as constants.
- `Buffer` is memory the CPU and the GPU share (Apple silicon has one
  memory), so feeding an input or reading a result is a plain store or load.
  A buffer that's one run's output and a later run's input keeps state on the
  GPU between runs.

`apple/nn.h` builds layers from these ops, each computing what its CPU
counterpart computes, so a model can run on either: `linear` (weights in the
CPU layout, `[out, in]`), `masked_layer_norm`, `attention` with padded keys,
`bce_logits` and `clipped_sgd`.

## Parameters

`apple/params.h` puts a CPU model's parameters on the GPU, one graph input
per tensor in `zf::visit_params` order. The CPU model stays the master copy
(save, load, evaluation). There are two ways to step:

- **The CPU steps.** `upload()` the weights before a run, and `add_grads()`
  the gradients the run computed into the model's.
- **The GPU steps.** `upload()` once, then run steps that read the parameters
  from one set of buffers and write the updated ones into a second set.
  `flip()` makes the second current, and `download()` at the end.

## LowResCross on the GPU

`apple/lowrescross_apple.h` runs [LowResCross](lowrescross.md) behind its
facade. The facade's weights stay the master copy. The GPU takes a copy only
when their version has moved, about once per step. The gradients add up on
the GPU over a sequence of runs and come back into the facade's once, at the
end of a batch's backward or at `backward_memories()`. It builds four graphs,
each for a capacity (passes and cache rows round up to powers of two):

| Graph  | For                                                                 |
|--------|---------------------------------------------------------------------|
| batch  | `forward_batch`: each pass's memory side and the stack, in one       |
| step   | `step_batch`: the batch graph with the model's heads and loss on top |
| memory | the memories' cache: rows in, every layer's cross K and V out        |
| pass   | `forward_shared`: the stack over K and V gathered from the cache     |

## A whole step: `LowResCross::step_batch`

On the CPU path a training step is `forward_batch()`, then the model's heads
and loss on the CPU, then `backward_batch()`. On the GPU that's two runs with
a wait between them, and the backward run computes the forward again.
`step_batch(batch, heads)` does it in one run: the stack, the heads, their
loss and every gradient. The stack's gradients add into its weights'
gradients and the heads' into theirs; the step itself is still the caller's.

The heads are the model's own code, built from graph ops:

```cpp
struct MyHeads {
  template <typename Visit> void visit_layers(Visit&&);   // the heads' layers
  struct Nodes { zf::apple::Node labels, logits; };       // what build() keeps
  zf::apple::Node build(zf::LowResHeadsGraph& hg, Nodes* n);
      // on hg.hidden [batch * q, kState] and hg.rows [batch * m, kWidth]:
      // parameters with hg.params.next(), inputs with hg.in.add(),
      // results with hg.out.add(); returns the loss, summed over passes
  void fill(const Nodes&, zf::LowResGraphInputs& in);     // labels, over zeros
  void read(const Nodes&, const zf::LowResGraphOutputs& out);  // e.g. logits
};

if (cross.steps_on_gpu(batch.size())) cross.step_batch(batch, heads);
else { /* forward_batch, the heads on the CPU, backward_batch */ }
```

- Padded passes get zero inputs, and the loss must weight them to nothing.
- After `step_batch()` there are no readouts to read, and `backward_batch()`
  throws.
- Each kind of heads gets its own graph per capacity.
- The test `LowResCross.AStepOnTheGpuMatchesTheCpu` checks a step against the
  CPU path: a linear head with squared error and a pointer with cross
  entropy.
