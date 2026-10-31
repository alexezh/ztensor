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

Batching does not make a single thread faster at these sizes. Time per example
is flat from a batch size of 1 to 64, because small layers already run in
4-row blocks. Batching is there so gradients can be averaged over many
examples and a later multi-threaded trainer can split a batch across cores.

