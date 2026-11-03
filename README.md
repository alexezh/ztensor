# ztensor

For my prototypes, I needed a C++ tensor library for both training and inference. 
My prototype models are small (millions of params) so there is no need in GPU. 

I looked at top projects with similar code and they were too big; and I did the common thing nowadays - vibe-coded 
a small C++20 tensor and transformer library. ZTenzor is fixed-size tensors, hand-written
forward and backward passes, plain SGD, and no dependencies beyond the standard
library. Shapes are template parameters, so a mismatched layer is a compile
error rather than a runtime one.

One interesting thing is that since lib is small and does not have dependencies, you can run your model in WASM. 

## Build

```sh
cmake --preset release        # Ninja + vcpkg; needs VCPKG_ROOT (GTest for tests)
cmake --build build/release
./build/release/ztensor_tests
./build/release/find_symbol   # the example below
```

Pass `-DZTENSOR_BUILD_TESTS=OFF` to build without GTest.

Other options:

- `-DZTENSOR_NATIVE=ON` compiles for the build machine's CPU (`-march=native`).
  It matters on x86, where the default baseline is SSE2; arm64's baseline
  already has NEON.
- `-DZTENSOR_ACCELERATE=ON` (the default on Apple) sends float stacks with
  enough work (at least 4 rows and 128K multiply-adds, `blas::use_gemm`)
  through Accelerate's SGEMM. On Apple Silicon it makes a 64-example decoder
  step about 1.4x faster; narrow single sequences keep the inline kernels.
  The GEMM sums in its own order, so batched results can differ from one
  example at a time in the last bits. Turn it off for bit-exact batching.
- `-DZTENSOR_CHECK_FINITE=ON` is a debug check. It throws
  `std::runtime_error` at the first NaN or infinity after any block's forward
  or backward pass. `zf::all_finite(t)` is available for your own checks.

## Example: a cross-attention decoder that learns to point

In this task, four memory tokens hold random symbols, and one query names a
symbol. The model has to answer which memory position holds that symbol. The
memory stays fixed for the whole pass; only the query is updated, block by
block. The full program is in `examples/find_symbol.cpp`. It reaches 100% in
about 8k steps, in under a second.

```cpp
#include "cross_attention.h"
#include "layers.h"
#include "loss.h"

constexpr std::size_t kVocab = 8, kMem = 4, kQuery = 1, kDim = 16, kFf = 32;

zf::Embedding<float, kVocab, kDim> symbol;   // memory tokens
zf::Embedding<float, kMem, kDim>   position; // lets the head name a slot
zf::Embedding<float, kVocab, kDim> ask;      // the query token
zf::TransformerDecoder<float, kQuery, kMem, kDim, kFf> decoder(/*levels=*/2, /*heads=*/2);
zf::Linear<float, kDim, kMem> head;          // logits over memory positions
zf::SoftmaxCrossEntropyLoss<float, kMem> loss;
// randomize each; embeddings with stddev 0.5, since the default 0.02 is too small to start

// --- one training step ---
zf::Tensor<float, kMem, kDim> memory;        // memory(m) = symbol[ids[m]] + position[m]
zf::Tensor<float, kQuery, kDim> query;       // query(0)  = ask[ids[answer]]

auto h = decoder.forward(query, memory);     // kQuery x kDim
float l = loss.forward(head.forward(h0), answer);  // h0 = row 0 of h

// backward: zero_grad() everything, then
auto d_h0 = head.backward(h0, loss.backward());
zf::Tensor<float, kMem, kDim> d_memory;      // gradients flow into the memory too
auto d_query = decoder.backward(d_h, &d_memory);   // d_h = d_h0 placed in row 0
ask.backward(ids[answer], row(d_query, 0));
for (m...) { symbol.backward(ids[m], row(d_memory, m)); position.backward(m, row(d_memory, m)); }
// apply_sgd(0.02f) on everything
```

Each `DecoderBlock` uses pre-norm and residual connections. It runs three steps
in order: self-attention among the queries, then cross-attention from the
queries to the memory, then a feed-forward layer. `TransformerDecoder` stacks
the blocks and adds a final LayerNorm. For a plain encoder over a single
sequence, use `Transformer<T, SeqLen, ModelDim, FfHidden>(depth, heads)` in the
same way.

## Activation and batching

The feed-forward activation is a template parameter that comes after
`FfHidden` and defaults to GELU:

```cpp
zf::TransformerDecoder<float, Q, M, D, F, zf::Activation::kRelu> decoder(4, 4);
```

GELU uses a vectorizable tanh accurate to within 3e-7 of `std::tanh`. The
forward pass stores the GELU derivative so the backward pass doesn't recompute
it.

The last template parameter, `Batch`, stacks that many independent examples
row by row. A batched tensor has `Batch * Q` rows, and example `b` occupies
rows `[b * Q, (b + 1) * Q)`. Attention is block-diagonal, so an example only
ever sees its own rows. Each example can have its own valid length, set with
`set_valid_length(b, n)` and `set_memory_length(b, n)`:

```cpp
zf::TransformerDecoder<float, Q, M, D, F, zf::Activation::kGelu, 64> decoder(4, 4);
decoder.zero_grad();
auto y = decoder.forward(x, memory);    // x: (64*Q) x D, memory: (64*M) x D
decoder.backward(d_y, &d_memory);       // gradients summed over the 64 examples
decoder.apply_sgd(lr / 64);             // one step for the whole batch
```

Tests check that a batch gives the same outputs and gradients as running its
examples one at a time.

Without Accelerate, batching doesn't make a single thread faster at these
sizes. Small layers already run in 4-row blocks, so time per example stays
flat from a batch size of 1 to 64. With Accelerate (the default on Apple), the
batch's projections become tall matrix multiplies and a 64-example step runs
about 1.4x faster. Batching also lets you average gradients over many examples,
and the threads below split a batch across cores.

## Threads

`zf::ThreadPool` (`thread_pool.h`) runs the library's data-parallel loops.
Work goes as items through one lock-free queue per worker; the caller works
too, and `Completion` tells it when its items are done. Idle workers spin
briefly, then sleep on an atomic.

```cpp
zf::parallel_for(&zf::ThreadPool::global(), rows, /*grain=*/32,
                 [&](std::size_t chunk, std::size_t begin, std::size_t end) {
                   // rows [begin, end); `chunk` indexes per-chunk partials
                 });
```

Chunks depend only on the size and the grain, never on the thread count, so
a loop that keeps per-chunk partial sums and adds them in chunk order gives
the same bits on one thread or twelve. What uses it:

- `LayerNorm::forward_rows` / `backward_rows` split a row stack.
- Large Linear calls on the inline kernels (no Accelerate, double, bf16)
  split their rows; backward sums dW per chunk.
- With Accelerate, a large GEMM splits into one part per matrix unit (one
  per performance cluster: two on an M2 Max), on a small pool of its own.
  More parts than units only contend: 224 x 354 -> 64 takes 21 us as one
  call, 13 us as two, 20 us or more as four or eight.

`ThreadPool::global()` takes most of the machine: on Apple the performance
cores less one (efficiency cores make stragglers in a fork-join loop;
workers run at the user-interactive QoS class); elsewhere every CPU the
process may use, less one. `ZF_THREADS=n` overrides it, and `ZF_THREADS=1`
keeps everything on the calling thread.

## bf16

Every layer, model and loss takes its element type as the first template
parameter. Both `float` and `zf::bf16` work:

```cpp
zf::TransformerDecoder<zf::bf16, Q, M, D, F> decoder(4, 4);
zf::Tensor<zf::bf16, Q, D> x = zf::tensor_cast<zf::bf16>(x_float);
auto y = decoder.forward(x, memory);    // y is bf16 too
decoder.apply_sgd(0.02f);               // learning rate stays float
```

`bf16` is a storage type. Weights, activations and the gradients passed
between layers are 16 bits. All arithmetic runs in float
(`zf::accumulator_t<bf16>`): sums, softmax and attention weights, layer-norm
statistics, losses, and parameter gradients (`weight_grad()` returns a float
tensor). A value is rounded to bf16, to nearest even, only where it is
stored. Reading a `bf16` gives a `float` implicitly. Writing one is explicit
(`bf16(x)` or `static_cast<T>(x)`), and `bf16` has no arithmetic operators of
its own, so every rounding point in the code is visible.

Accuracy: a bf16 model matches the same model in float to within about 1% per
layer and a few percent through a stack. Trained end to end in bf16, the
pointing example above reaches 100% accuracy, the same as float.

Speed: there's no bf16 arithmetic on the CPU without the BF16 instructions,
so bf16 is for memory, not time. A 48-row `Linear` is about 1.5x slower than
float, mostly the cost of rounding its output. A 64-example decoder step is
about 1.2x slower. Stacks of 8 or more rows widen the weights to float once
and run the float kernels, and tall stacks go through Accelerate.

