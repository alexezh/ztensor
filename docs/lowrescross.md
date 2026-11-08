# LowResCross

`lowrescross.h`: a cross transformer over wide, sparse rows, computed at low
width.

Every query row and every memory row is one wide vector (`Layout::kWidth`),
mostly absent for any one row. A read mask says which slots are present.
Part of the row is state (`Layout::kState` slots at `Layout::kStateOffset`),
the residual stream the layers write; the rest stays fixed through the stack.
Each layer reads the whole row through a norm over its present slots,
projects it down to the width it computes in, and restores the result into
the state, only where the row's write mask allows:

```
self   h += mask · O_s(attn(Q_s x̂, K_s x̂, V_s x̂))     among the queries
cross  h += mask · O_c(attn(Q_c x̂', K_c m̂, V_c m̂))   into the memory
ffn    h += mask · W_2 relu(W_1 x̂'')                    within the row
```

Attention works in `Heads x HeadDim` and the feed-forward in `Ffn`, small
enough to keep the transformer cheap however wide the row. The memory is
fixed for the pass (decoder-only).

```cpp
zf::LowResCross<float, MyLayout, Q, M, /*Heads=*/4, /*HeadDim=*/8, /*Ffn=*/32> cross(/*levels=*/2);
cross.randomize(rng);
const auto& h = cross.forward(queries, memory);   // Q x kState readouts
auto d_state = cross.backward(d_h);               // per query row
cross.apply_sgd(lr);
```

## The layout

LowResCross knows nothing of what a row means. The model supplies that as a
`Layout` (`lowrescross_layout.h`): the row's `Input` type, its widths, a few
learned id slots (each looked up in its own embedding table and added into
the row), and how to write a row, its read mask and its write mask. It also
says how to hand back the gradient on a query row's input state
(`StateGrad`), so a model can train through state it wrote in an earlier
pass.

## Batches

`forward_batch()` runs many passes at once: different examples, each with its
own queries and memory, that never see each other. A pass is indexed by its
place in the batch. `readout_of(b)` gives pass `b`'s readouts, and
`backward_batch()` takes one gradient per pass.

## The memories' cache

Many passes over the same few memories read memories the implementation
keeps. A memory row's cross K and V depend on that row alone, so each row goes
through the memory side once, when it is added:

```cpp
cross.set_memories({&mem0, &mem1});
cross.add_memory(mem2);                         // returns its index
cross.remove_memory_rows(0, {3});               // the rest keep their order
cross.forward_shared({&q0, &q1}, {0, 2});       // pass b reads memory_of[b]
cross.backward_batch(d);                        // again, as often as needed
cross.backward_memories();                      // the memories' gradients, once
```

The memories' gradients add up over every `backward_batch()` until
`backward_memories()` takes them back through once. The memories can't change
in between, nor can the weights. When the weights do change (a step), the
cache's K and V are made again from them before the next pass reads them.

## The pointer

`LowResPointer<T, Layout, M, K>` scores a query row's readout against every
memory row of its pass: `logit_j = (A · readout) · (B · memory_j)`. It's
order-free. Its gradient on the memory rows goes to the cross through
`memory_grad(b)`.

## Where it runs

`LowResCross` is a facade. It holds the weights, and an implementation runs
them: `lowrescross_cpu.h`, or `apple/lowrescross_apple.h` on the GPU. Saving,
loading and the optimizer see one model whatever ran it.

- `set_device(LowResDevice::kCpu | kApple | kAuto)` picks where batches run.
  `kAuto` (the default) sends a batch of 64 passes or more
  (`kLowResGpuMinPasses`) to the GPU when it's on, and keeps anything smaller
  on the CPU.
- The GPU is on when the environment variable `ZTENSOR_GPU` is 1, off when
  it's 0, and otherwise follows `ZTENSOR_LOWRES_GPU_DEFAULT` (0 unless the
  program defines it). A program can rename the variable with
  `ZTENSOR_LOWRES_GPU_ENV`.
- `forward()`, a batch of one, always runs on the CPU.

The weights carry a version that `randomize()`, `apply_sgd()` and
`visit_layers()` move. The GPU copies the weights only when the version has
moved, so a model that runs many batches in a step uploads its weights once.
The whole step can also run in one GPU run, the model's heads included; see
[Apple GPU](apple_gpu.md#a-whole-step-lowrescrossstep_batch).
