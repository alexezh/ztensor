// Microbenchmarks for the tensor hot paths. Run before and after any change to
// the matrix logic: the numbers here are the only reason to prefer one loop
// over another.
#include <chrono>
#include <memory>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "attention.h"
#include "cross_attention.h"
#include "layers.h"
#include "tensor.h"
#include "transformer.h"

namespace {

using Clock = std::chrono::steady_clock;

template <typename Body>
double time_ms(const char* what, std::size_t reps, Body&& body) {
  // One untimed pass, so a first-touch page fault is not the measurement.
  body();
  const auto start = Clock::now();
  for (std::size_t i = 0; i < reps; ++i) {
    body();
  }
  const auto end = Clock::now();
  const double total =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::printf("  %-44s %9.3f ms  %9.1f ns/op\n", what, total,
              1e6 * total / static_cast<double>(reps));
  return total;
}

template <typename Tensor>
void fill(Tensor& t, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.5f);
  for (std::size_t i = 0; i < t.size(); ++i) {
    t.data()[i] = dist(rng);
  }
}

// A sink the optimizer cannot see through, so a measured construction is
// really performed instead of being elided.
volatile float g_sink = 0.0f;

// The shapes the ZDOG models actually use.
constexpr std::size_t kSeq = 48;
constexpr std::size_t kDim = 32;
constexpr std::size_t kFf = 64;
constexpr std::size_t kQueries = 13;

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::printf("tensor hot paths, shapes as used by ZVisionComp/ZForward\n");
  std::printf("  sequence %zu, model dim %zu, ffn %zu\n\n", kSeq, kDim, kFf);

  // --- allocation: what a temporary costs before any arithmetic happens ---
  std::printf("allocation\n");
  time_ms("construct Tensor<float,48,32> (zeroed)", 200000, [&] {
    zf::Tensor<float, kSeq, kDim> t;
    g_sink += t.data()[kSeq * kDim - 1];
  });
  time_ms("construct Tensor<float,32>", 200000, [&] {
    zf::Tensor<float, kDim> t;
    g_sink += t.data()[kDim - 1];
  });
  {
    zf::Tensor<float, kSeq, kDim> source;
    fill(source, rng);
    time_ms("copy Tensor<float,48,32>", 200000, [&] {
      zf::Tensor<float, kSeq, kDim> copy = source;
      g_sink += copy.data()[kSeq * kDim - 1];
    });
  }
  std::printf("  (a 48x32 tensor is %zu bytes)\n",
              sizeof(float) * kSeq * kDim);

  // --- Linear, the single matmul everything is built from ---
  std::printf("\nLinear<32,32>\n");
  {
    zf::Linear<float, kDim, kDim> layer;
    layer.randomize(rng);
    zf::Tensor<float, kSeq, kDim> x;
    zf::Tensor<float, kSeq, kDim> d_out;
    fill(x, rng);
    fill(d_out, rng);
    time_ms("forward, 48 rows", 20000,
            [&] { volatile float v = layer.template forward<kSeq>(x)(0, 0); (void)v; });
    time_ms("backward, 48 rows", 20000, [&] {
      volatile float v = layer.template backward<kSeq>(x, d_out)(0, 0);
      (void)v;
    });
    zf::Tensor<float, kDim> xv;
    zf::Tensor<float, kDim> dv;
    fill(xv, rng);
    fill(dv, rng);
    time_ms("forward, single vector", 500000,
            [&] { volatile float v = layer.forward(xv)(0); (void)v; });
    time_ms("backward, single vector", 500000,
            [&] { volatile float v = layer.backward(xv, dv)(0); (void)v; });
  }

  std::printf("\nLinear<32,64> and <64,32>, the feed-forward pair\n");
  {
    zf::Linear<float, kDim, kFf> up;
    zf::Linear<float, kFf, kDim> down;
    up.randomize(rng);
    down.randomize(rng);
    zf::Tensor<float, kSeq, kDim> x;
    zf::Tensor<float, kSeq, kFf> h;
    fill(x, rng);
    fill(h, rng);
    time_ms("up forward, 48 rows", 20000,
            [&] { volatile float v = up.template forward<kSeq>(x)(0, 0); (void)v; });
    time_ms("down forward, 48 rows", 20000,
            [&] { volatile float v = down.template forward<kSeq>(h)(0, 0); (void)v; });
  }

  // --- attention ---
  std::printf("\nMultiHeadSelfAttention<48,32>, 4 heads\n");
  {
    zf::MultiHeadSelfAttention<float, kSeq, kDim> attention(4);
    attention.randomize(rng);
    zf::Tensor<float, kSeq, kDim> x;
    zf::Tensor<float, kSeq, kDim> d_out;
    fill(x, rng);
    fill(d_out, rng);
    time_ms("forward, full sequence", 5000,
            [&] { volatile float v = attention.forward(x)(0, 0); (void)v; });
    time_ms("backward, full sequence", 5000,
            [&] { volatile float v = attention.backward(d_out)(0, 0); (void)v; });
    // What the models really do: a padded batch where only part of the
    // sequence is real.
    attention.set_valid_length(24);
    time_ms("forward, valid_length 24 of 48", 5000,
            [&] { volatile float v = attention.forward(x)(0, 0); (void)v; });
    time_ms("backward, valid_length 24 of 48", 5000,
            [&] { volatile float v = attention.backward(d_out)(0, 0); (void)v; });
  }

  std::printf("\nMultiHeadCrossAttention<13,48,32>, 4 heads\n");
  {
    zf::MultiHeadCrossAttention<float, kQueries, kSeq, kDim> cross(4);
    cross.randomize(rng);
    zf::Tensor<float, kQueries, kDim> q;
    zf::Tensor<float, kSeq, kDim> memory;
    zf::Tensor<float, kQueries, kDim> d_out;
    fill(q, rng);
    fill(memory, rng);
    fill(d_out, rng);
    time_ms("forward", 20000,
            [&] { volatile float v = cross.forward(q, memory)(0, 0); (void)v; });
    time_ms("backward", 20000, [&] {
      zf::Tensor<float, kSeq, kDim> d_memory;
      volatile float v = cross.backward(d_out, &d_memory)(0, 0);
      (void)v;
    });
  }

  // --- whole blocks, which is what a training step is made of ---
  std::printf("\nblocks\n");
  {
    zf::TransformerBlock<float, kSeq, kDim, kFf> block(4);
    block.randomize(rng);
    zf::Tensor<float, kSeq, kDim> x;
    zf::Tensor<float, kSeq, kDim> d_out;
    fill(x, rng);
    fill(d_out, rng);
    time_ms("TransformerBlock forward", 5000,
            [&] { volatile float v = block.forward(x)(0, 0); (void)v; });
    time_ms("TransformerBlock forward+backward", 2000, [&] {
      block.forward(x);
      volatile float v = block.backward(d_out)(0, 0);
      (void)v;
    });
  }
  {
    zf::DecoderBlock<float, kQueries, kSeq, kDim, kFf> block(4);
    block.randomize(rng);
    zf::Tensor<float, kQueries, kDim> x;
    zf::Tensor<float, kSeq, kDim> memory;
    zf::Tensor<float, kQueries, kDim> d_out;
    fill(x, rng);
    fill(memory, rng);
    fill(d_out, rng);
    time_ms("DecoderBlock forward", 10000,
            [&] { volatile float v = block.forward(x, memory)(0, 0); (void)v; });
    time_ms("DecoderBlock forward+backward", 5000, [&] {
      block.forward(x, memory);
      zf::Tensor<float, kSeq, kDim> d_memory;
      volatile float v = block.backward(d_out, &d_memory)(0, 0);
      (void)v;
    });
  }

  // --- batching: the ZAction decoder shape, 64 examples per SGD step ---
  //
  // 4 levels, 4 heads, 12 queries over 12 memory tokens, d 32, FFN 64.
  // "one at a time" is 64 forward+backward passes of a Batch=1 decoder;
  // "one batch" is a single pass of the same decoder with Batch=64.
  std::printf("\nZAction decoder, 64 examples (4 levels, 12 q x 12 mem, d 32)\n");
  {
    constexpr std::size_t kQ = 12, kM = 12, kB = 64;
    zf::TransformerDecoder<float, kQ, kM, kDim, kFf> one(4, 4);
    one.randomize(rng);
    zf::Tensor<float, kQ, kDim> x;
    zf::Tensor<float, kM, kDim> memory;
    zf::Tensor<float, kQ, kDim> d_out;
    fill(x, rng);
    fill(memory, rng);
    fill(d_out, rng);
    const double t1 = time_ms("one at a time (64 x Batch=1)", 50, [&] {
      for (std::size_t b = 0; b < kB; ++b) {
        one.forward(x, memory);
        zf::Tensor<float, kM, kDim> d_memory;
        volatile float v = one.backward(d_out, &d_memory)(0, 0);
        (void)v;
      }
    });

    auto batched = std::make_unique<zf::TransformerDecoder<float, kQ, kM, kDim, kFf, kB>>(4, 4);
    batched->randomize(rng);
    zf::Tensor<float, kB * kQ, kDim> xb;
    zf::Tensor<float, kB * kM, kDim> memb;
    zf::Tensor<float, kB * kQ, kDim> d_outb;
    fill(xb, rng);
    fill(memb, rng);
    fill(d_outb, rng);
    const double t2 = time_ms("one batch (Batch=64)", 50, [&] {
      batched->forward(xb, memb);
      zf::Tensor<float, kB * kM, kDim> d_memory;
      volatile float v = batched->backward(d_outb, &d_memory)(0, 0);
      (void)v;
    });
    std::printf("  batch speedup %.2fx\n", t1 / t2);
  }
  return 0;
}
