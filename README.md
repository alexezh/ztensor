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

## Docs

- [Core](docs/core.md): build options, threads, data-parallel batches, bf16
- [Transformer](docs/transformer.md): encoder and decoder blocks, activation, batching
- [LowResCross](docs/lowrescross.md): a cross transformer over wide, sparse rows
- [Apple GPU](docs/apple_gpu.md): training graphs on Metal (MPSGraph)
- [Reinforcement learning](docs/rl.md): RLTrainer, policy-gradient rounds
