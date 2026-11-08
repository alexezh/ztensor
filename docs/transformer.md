# Transformer

`transformer.h` and `cross_attention.h`: the encoder and decoder stacks, with
hand-written forward and backward passes. The README's example uses
`TransformerDecoder`; the full program is `examples/find_symbol.cpp`.

## Blocks

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
and the threads ([Core](core.md#threads)) split a batch across cores.
