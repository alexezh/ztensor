#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "cross_attention.h"
#include "grad_check.h"
#include "tensor.h"

namespace {

constexpr std::size_t kQ = 3;
constexpr std::size_t kM = 5;
constexpr std::size_t kD = 4;
constexpr std::size_t kFf = 8;

using Cross = zf::MultiHeadCrossAttention<float, kQ, kM, kD>;
using QuerySeq = Cross::QuerySequence;
using MemorySeq = Cross::MemorySequence;

template <typename Tensor>
void fill(Tensor& t, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.7f);
  for (std::size_t i = 0; i < t.size(); ++i) {
    t.data()[i] = dist(rng);
  }
}

TEST(CrossAttention, RejectsAHeadCountThatDoesNotDivideTheModel) {
  EXPECT_THROW((Cross(3)), std::invalid_argument);
  EXPECT_THROW((Cross(0)), std::invalid_argument);
  EXPECT_NO_THROW((Cross(2)));
}

TEST(CrossAttention, RejectsAnEmptyMemory) {
  Cross attention(2);
  EXPECT_THROW(attention.set_memory_length(0), std::out_of_range);
  EXPECT_THROW(attention.set_memory_length(kM + 1), std::out_of_range);
  EXPECT_NO_THROW(attention.set_memory_length(kM));
}

TEST(CrossAttention, EveryQueryAttendsOverTheWholeMemory) {
  Cross attention(2);
  std::mt19937 rng(1);
  attention.randomize(rng);
  QuerySeq q;
  MemorySeq memory;
  fill(q, rng);
  fill(memory, rng);
  attention.forward(q, memory);

  // Unlike self attention there is no diagonal and no causal structure: a
  // query may read any memory position, and the weights are a distribution
  // over the memory.
  for (std::size_t h = 0; h < attention.heads(); ++h) {
    const auto& weights = attention.attention_weights(h);
    for (std::size_t i = 0; i < kQ; ++i) {
      float sum = 0.0f;
      for (std::size_t j = 0; j < kM; ++j) {
        EXPECT_GE(weights(i, j), 0.0f);
        sum += weights(i, j);
      }
      EXPECT_NEAR(sum, 1.0f, 1e-5);
    }
  }
}

TEST(CrossAttention, MaskedMemoryPositionsAreNeverRead) {
  Cross attention(2);
  std::mt19937 rng(2);
  attention.randomize(rng);
  QuerySeq q;
  MemorySeq memory;
  fill(q, rng);
  fill(memory, rng);
  attention.set_memory_length(3);
  const QuerySeq out = attention.forward(q, memory);

  // Change the padded tail of the memory; the output must not budge. This is
  // what lets one compile-time memory length hold scenes of different sizes.
  for (std::size_t j = 3; j < kM; ++j) {
    for (std::size_t d = 0; d < kD; ++d) {
      memory(j, d) += 10.0f;
    }
  }
  const QuerySeq again = attention.forward(q, memory);
  for (std::size_t i = 0; i < kQ; ++i) {
    for (std::size_t d = 0; d < kD; ++d) {
      EXPECT_FLOAT_EQ(out(i, d), again(i, d));
    }
  }
}

// The gradient checks. Hand-written backprop that is only ever exercised by a
// training loop will happily converge to something with a sign slip in it, so
// every path is compared against central differences.
TEST(CrossAttention, QueryGradientMatchesFiniteDifferences) {
  Cross attention(2);
  std::mt19937 rng(3);
  attention.randomize(rng);
  QuerySeq q;
  MemorySeq memory;
  fill(q, rng);
  fill(memory, rng);

  // loss = sum of the output, so dL/dout is all ones.
  const auto loss = [&]() {
    const QuerySeq out = attention.forward(q, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      sum += out.data()[i];
    }
    return sum;
  };
  loss();
  QuerySeq d_out;
  d_out.fill(1.0f);
  MemorySeq d_memory;
  const QuerySeq d_q = attention.backward(d_out, &d_memory);

  std::vector<float*> slots;
  std::vector<double> analytic;
  for (std::size_t i = 0; i < q.size(); ++i) {
    slots.push_back(&q.data()[i]);
    analytic.push_back(static_cast<double>(d_q.data()[i]));
  }
  zf_test::expect_grads_match(slots, analytic, loss, "cross attention dq");
}

TEST(CrossAttention, MemoryGradientMatchesFiniteDifferences) {
  Cross attention(2);
  std::mt19937 rng(4);
  attention.randomize(rng);
  QuerySeq q;
  MemorySeq memory;
  fill(q, rng);
  fill(memory, rng);

  const auto loss = [&]() {
    const QuerySeq out = attention.forward(q, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      sum += out.data()[i];
    }
    return sum;
  };
  loss();
  QuerySeq d_out;
  d_out.fill(1.0f);
  MemorySeq d_memory;
  attention.backward(d_out, &d_memory);

  std::vector<float*> slots;
  std::vector<double> analytic;
  for (std::size_t i = 0; i < memory.size(); ++i) {
    slots.push_back(&memory.data()[i]);
    analytic.push_back(static_cast<double>(d_memory.data()[i]));
  }
  zf_test::expect_grads_match(slots, analytic, loss,
                              "cross attention d_memory");
}

TEST(CrossAttention, MemoryGradientAccumulatesAcrossCalls) {
  // Every decoder block reads the same memory, so the encoder's gradient is a
  // sum over blocks. backward() adds into the output rather than assigning it.
  Cross attention(2);
  std::mt19937 rng(5);
  attention.randomize(rng);
  QuerySeq q;
  MemorySeq memory;
  fill(q, rng);
  fill(memory, rng);
  attention.forward(q, memory);

  QuerySeq d_out;
  d_out.fill(1.0f);
  MemorySeq once;
  attention.backward(d_out, &once);
  MemorySeq twice;
  attention.backward(d_out, &twice);
  attention.backward(d_out, &twice);
  for (std::size_t i = 0; i < once.size(); ++i) {
    EXPECT_NEAR(2.0f * once.data()[i], twice.data()[i], 1e-4);
  }
}

TEST(CrossAttention, ToleratesANullMemoryGradient) {
  // A caller that only wants dL/d(queries) passes nullptr.
  Cross attention(2);
  std::mt19937 rng(6);
  attention.randomize(rng);
  QuerySeq q;
  MemorySeq memory;
  fill(q, rng);
  fill(memory, rng);
  attention.forward(q, memory);
  QuerySeq d_out;
  d_out.fill(1.0f);
  EXPECT_NO_THROW(attention.backward(d_out, nullptr));
}

// --- the decoder block and stack ------------------------------------------

using Block = zf::DecoderBlock<float, kQ, kM, kD, kFf>;
using Decoder = zf::TransformerDecoder<float, kQ, kM, kD, kFf>;

TEST(DecoderBlock, QueryGradientMatchesFiniteDifferences) {
  Block block(2);
  std::mt19937 rng(7);
  block.randomize(rng);
  QuerySeq x;
  MemorySeq memory;
  fill(x, rng);
  fill(memory, rng);

  const auto loss = [&]() {
    const QuerySeq out = block.forward(x, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      sum += out.data()[i];
    }
    return sum;
  };
  loss();
  QuerySeq d_out;
  d_out.fill(1.0f);
  MemorySeq d_memory;
  const QuerySeq dx = block.backward(d_out, &d_memory);

  std::vector<float*> slots;
  std::vector<double> analytic;
  for (std::size_t i = 0; i < x.size(); ++i) {
    slots.push_back(&x.data()[i]);
    analytic.push_back(static_cast<double>(dx.data()[i]));
  }
  zf_test::expect_grads_match(slots, analytic, loss, "decoder block dx");
}

TEST(DecoderBlock, MemoryGradientMatchesFiniteDifferences) {
  Block block(2);
  std::mt19937 rng(8);
  block.randomize(rng);
  QuerySeq x;
  MemorySeq memory;
  fill(x, rng);
  fill(memory, rng);

  const auto loss = [&]() {
    const QuerySeq out = block.forward(x, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      sum += out.data()[i];
    }
    return sum;
  };
  loss();
  QuerySeq d_out;
  d_out.fill(1.0f);
  MemorySeq d_memory;
  block.backward(d_out, &d_memory);

  std::vector<float*> slots;
  std::vector<double> analytic;
  for (std::size_t i = 0; i < memory.size(); ++i) {
    slots.push_back(&memory.data()[i]);
    analytic.push_back(static_cast<double>(d_memory.data()[i]));
  }
  zf_test::expect_grads_match(slots, analytic, loss, "decoder block d_memory");
}

TEST(TransformerDecoder, GradientsMatchFiniteDifferencesThroughTheStack) {
  Decoder decoder(2, 2);
  std::mt19937 rng(9);
  decoder.randomize(rng);
  QuerySeq x;
  MemorySeq memory;
  fill(x, rng);
  fill(memory, rng);

  const auto loss = [&]() {
    const QuerySeq out = decoder.forward(x, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      sum += out.data()[i];
    }
    return sum;
  };
  loss();
  QuerySeq d_out;
  d_out.fill(1.0f);
  MemorySeq d_memory;
  const QuerySeq dx = decoder.backward(d_out, &d_memory);

  std::vector<float*> query_slots;
  std::vector<double> query_analytic;
  for (std::size_t i = 0; i < x.size(); ++i) {
    query_slots.push_back(&x.data()[i]);
    query_analytic.push_back(static_cast<double>(dx.data()[i]));
  }
  zf_test::expect_grads_match(query_slots, query_analytic, loss, "decoder dx");

  std::vector<float*> memory_slots;
  std::vector<double> memory_analytic;
  for (std::size_t i = 0; i < memory.size(); ++i) {
    memory_slots.push_back(&memory.data()[i]);
    memory_analytic.push_back(static_cast<double>(d_memory.data()[i]));
  }
  zf_test::expect_grads_match(memory_slots, memory_analytic, loss,
                              "decoder d_memory");
}

TEST(TransformerDecoder, DepthIsRuntimeAndAtLeastOne) {
  EXPECT_THROW((Decoder(0, 2)), std::invalid_argument);
  EXPECT_EQ(Decoder(1, 2).depth(), 1u);
  EXPECT_EQ(Decoder(4, 2).depth(), 4u);
  EXPECT_EQ(Decoder(4, 2).heads(), 2u);
}

TEST(TransformerDecoder, OneStepOfDescentReducesTheLoss) {
  Decoder decoder(2, 2);
  std::mt19937 rng(10);
  decoder.randomize(rng);
  QuerySeq x;
  MemorySeq memory;
  fill(x, rng);
  fill(memory, rng);

  // Push the output towards zero and check the step helps.
  const auto objective = [&]() {
    const QuerySeq out = decoder.forward(x, memory);
    float sum = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      sum += out.data()[i] * out.data()[i];
    }
    return sum;
  };
  const float before = objective();
  QuerySeq out = decoder.forward(x, memory);
  QuerySeq d_out;
  for (std::size_t i = 0; i < out.size(); ++i) {
    d_out.data()[i] = 2.0f * out.data()[i];
  }
  decoder.zero_grad();
  MemorySeq d_memory;
  decoder.backward(d_out, &d_memory);
  decoder.apply_sgd(0.01f);
  EXPECT_LT(objective(), before);
}

}  // namespace
