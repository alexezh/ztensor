# Core

## Build options

Besides `-DZTENSOR_BUILD_TESTS=OFF` (no GTest):

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
- `-DZTENSOR_APPLE_GPU=ON` (the default on Apple) builds `ztensor_apple`, the
  GPU graphs; see [Apple GPU](apple_gpu.md).

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

## Data-parallel batches

`zf::Replicas<Model>` (`replicas.h`) splits a batch of examples into slices
of whole examples. Slice 0 runs on the model itself, the others on helper
copies made on first use; each slice runs forward and backward on its own
copy, so the slices share nothing while they run. Then two passes over the
parameters, split by tensor over the pool:

- reduce: the model's gradient += the helpers', in slice order, and the
  squared norm of the sum;
- update: `w -= rate(norm²) · g`, then the new weights are copied into every
  helper.

The step size is the caller's: `rate` gets the summed gradient's squared norm
(for clipping) and returns the factor to step by. The slices depend only on
the batch size and the slice count, never on the number of threads, so the
result is the same on one thread as on twelve. Slicing costs work (each pass
has a fixed cost), so it pays only when the threads would otherwise idle.

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
pointing example in the README reaches 100% accuracy, the same as float.

Speed: there's no bf16 arithmetic on the CPU without the BF16 instructions,
so bf16 is for memory, not time. A 48-row `Linear` is about 1.5x slower than
float, mostly the cost of rounding its output. A 64-example decoder step is
about 1.2x slower. Stacks of 8 or more rows widen the weights to float once
and run the float kernels, and tall stacks go through Accelerate.
