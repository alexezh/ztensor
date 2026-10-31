// A cross-attention decoder learns to point: given 4 memory tokens holding
// random symbols and one query naming a symbol, answer which memory position
// holds it.
#include <cstdio>
#include <random>

#include "cross_attention.h"
#include "layers.h"
#include "loss.h"

int main() {
  constexpr std::size_t kVocab = 8, kMem = 4, kQuery = 1, kDim = 16, kFf = 32;
  std::mt19937_64 rng(1);

  zf::Embedding<float, kVocab, kDim> symbol;  // memory tokens
  zf::Embedding<float, kMem, kDim> position;  // so the head can name a slot
  zf::Embedding<float, kVocab, kDim> ask;     // the query token
  zf::TransformerDecoder<float, kQuery, kMem, kDim, kFf> decoder(/*levels=*/2,
                                                                  /*heads=*/2);
  zf::Linear<float, kDim, kMem> head;  // logits over memory positions
  zf::SoftmaxCrossEntropyLoss<float, kMem> loss;
  symbol.randomize(rng, 0.5f);  // default 0.02 is too small to start
  position.randomize(rng, 0.5f);
  ask.randomize(rng, 0.5f);
  decoder.randomize(rng);
  head.randomize(rng);

  std::uniform_int_distribution<std::size_t> pick_symbol(0, kVocab - 1);
  std::uniform_int_distribution<std::size_t> pick_slot(0, kMem - 1);
  std::size_t correct = 0;
  for (int step = 1; step <= 20000; ++step) {
    std::size_t ids[kMem];
    for (std::size_t& id : ids) id = pick_symbol(rng);
    const std::size_t answer = pick_slot(rng);

    // forward: memory is fixed for the pass, only the query evolves
    zf::Tensor<float, kMem, kDim> memory;
    for (std::size_t m = 0; m < kMem; ++m) {
      const auto s = symbol.forward(ids[m]);
      const auto p = position.forward(m);
      for (std::size_t d = 0; d < kDim; ++d) memory(m, d) = s(d) + p(d);
    }
    zf::Tensor<float, kQuery, kDim> query;
    const auto q = ask.forward(ids[answer]);
    for (std::size_t d = 0; d < kDim; ++d) query(0, d) = q(d);

    const auto h = decoder.forward(query, memory);
    zf::Tensor<float, kDim> h0;
    for (std::size_t d = 0; d < kDim; ++d) h0(d) = h(0, d);
    loss.forward(head.forward(h0), answer);
    // a repeated symbol makes several slots right
    correct += ids[loss.argmax()] == ids[answer];

    // backward, then one SGD step
    symbol.zero_grad(); position.zero_grad(); ask.zero_grad();
    decoder.zero_grad(); head.zero_grad();
    const auto d_h0 = head.backward(h0, loss.backward());
    zf::Tensor<float, kQuery, kDim> d_h;
    for (std::size_t d = 0; d < kDim; ++d) d_h(0, d) = d_h0(d);
    zf::Tensor<float, kMem, kDim> d_memory;
    const auto d_query = decoder.backward(d_h, &d_memory);
    zf::Tensor<float, kDim> row;
    for (std::size_t d = 0; d < kDim; ++d) row(d) = d_query(0, d);
    ask.backward(ids[answer], row);
    for (std::size_t m = 0; m < kMem; ++m) {
      for (std::size_t d = 0; d < kDim; ++d) row(d) = d_memory(m, d);
      symbol.backward(ids[m], row);
      position.backward(m, row);
    }
    const float lr = 0.02f;
    symbol.apply_sgd(lr); position.apply_sgd(lr); ask.apply_sgd(lr);
    decoder.apply_sgd(lr); head.apply_sgd(lr);

    if (step % 4000 == 0) {
      std::printf("step %5d  accuracy %.3f\n", step, correct / 4000.0);
      correct = 0;
    }
  }
}
