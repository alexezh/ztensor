#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "attention.h"
#include "dot.h"
#include "grad_check.h"
#include "loss.h"
#include "transformer.h"

namespace {

using zf_test::expect_grads_match;
using zf_test::randomize_tensor;
using zf_test::slots_of;
using zf_test::values_of;

constexpr std::size_t kSeq = 4;
constexpr std::size_t kModel = 8;
constexpr std::size_t kFf = 16;

using Attention = zf::MultiHeadSelfAttention<float, kSeq, kModel>;
using Block = zf::TransformerBlock<float, kSeq, kModel, kFf>;
using Stack = zf::Transformer<float, kSeq, kModel, kFf>;

TEST(Attention, HeadCountMustDivideModelDim) {
  EXPECT_NO_THROW(Attention(1));
  EXPECT_NO_THROW(Attention(2));
  EXPECT_NO_THROW(Attention(4));
  EXPECT_NO_THROW(Attention(8));
  EXPECT_THROW(Attention(3), std::invalid_argument);
  EXPECT_THROW(Attention(5), std::invalid_argument);
  EXPECT_THROW(Attention(0), std::invalid_argument);
}

TEST(Attention, HeadCountIsRuntimeButShapesAreNot) {
  // Both head counts are the same C++ type: heads do not affect parameter
  // shapes, so they are a runtime knob.
  Attention one(1);
  Attention four(4);
  static_assert(std::is_same_v<decltype(one), decltype(four)>);
  EXPECT_EQ(one.head_dim(), kModel);
  EXPECT_EQ(four.head_dim(), kModel / 4);
  // 4 projections of 8x8 + bias
  static_assert(Attention::parameter_count == 4 * (8 * 8 + 8));
}

TEST(Attention, WeightsAreARowStochasticMatrix) {
  auto gen = zf_test::rng();
  Attention attn(2);
  attn.randomize(gen);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen);
  attn.forward(x);

  for (std::size_t h = 0; h < attn.heads(); ++h) {
    const auto& a = attn.attention_weights(h);
    for (std::size_t i = 0; i < kSeq; ++i) {
      float sum = 0.0f;
      for (std::size_t j = 0; j < kSeq; ++j) {
        EXPECT_GE(a(i, j), 0.0f);
        sum += a(i, j);
      }
      EXPECT_NEAR(sum, 1.0f, 1e-5f) << "head " << h << " row " << i;
    }
  }
}

TEST(Attention, CausalMaskZerosTheFuture) {
  auto gen = zf_test::rng();
  Attention attn(2, /*causal=*/true);
  attn.randomize(gen);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen);
  attn.forward(x);

  for (std::size_t h = 0; h < attn.heads(); ++h) {
    const auto& a = attn.attention_weights(h);
    for (std::size_t i = 0; i < kSeq; ++i) {
      float sum = 0.0f;
      for (std::size_t j = 0; j < kSeq; ++j) {
        if (j > i) {
          EXPECT_FLOAT_EQ(a(i, j), 0.0f) << "position " << i << " saw " << j;
        }
        sum += a(i, j);
      }
      EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
  }
}

TEST(Attention, CausalForwardIgnoresLaterTokens) {
  auto gen = zf_test::rng();
  Attention attn(2, /*causal=*/true);
  attn.randomize(gen);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen);

  const auto base = attn.forward(x);

  // Perturbing the last token must not change the first token's output.
  zf::TensorF<kSeq, kModel> perturbed = x;
  for (std::size_t d = 0; d < kModel; ++d) {
    perturbed(kSeq - 1, d) += 3.0f;
  }
  const auto after = attn.forward(perturbed);

  for (std::size_t d = 0; d < kModel; ++d) {
    EXPECT_NEAR(base(0, d), after(0, d), 1e-5f);
  }
  bool last_changed = false;
  for (std::size_t d = 0; d < kModel; ++d) {
    if (std::abs(base(kSeq - 1, d) - after(kSeq - 1, d)) > 1e-4f) {
      last_changed = true;
    }
  }
  EXPECT_TRUE(last_changed);
}

void CheckAttentionGradients(std::size_t heads, bool causal) {
  auto gen = zf_test::rng();
  Attention attn(heads, causal);
  attn.randomize(gen);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen, 0.7);
  zf::TensorF<kSeq, kModel> w;
  randomize_tensor(w, gen, 0.7);

  const auto loss = [&] { return zf::dot(attn.forward(x), w); };
  const std::string what =
      "attention h=" + std::to_string(heads) + (causal ? " causal" : "");

  attn.zero_grad();
  attn.forward(x);
  const auto dx = attn.backward(w);

  expect_grads_match(slots_of(attn.query_proj().weight()),
                     values_of(attn.query_proj().weight_grad()), loss,
                     what + " wq");
  expect_grads_match(slots_of(attn.key_proj().weight()),
                     values_of(attn.key_proj().weight_grad()), loss,
                     what + " wk");
  expect_grads_match(slots_of(attn.value_proj().weight()),
                     values_of(attn.value_proj().weight_grad()), loss,
                     what + " wv");
  expect_grads_match(slots_of(attn.output_proj().weight()),
                     values_of(attn.output_proj().weight_grad()), loss,
                     what + " wo");
  expect_grads_match(slots_of(x), values_of(dx), loss, what + " input");
}

TEST(Attention, GradientsMatchFiniteDifferencesSingleHead) {
  CheckAttentionGradients(1, false);
}

TEST(Attention, GradientsMatchFiniteDifferencesMultiHead) {
  CheckAttentionGradients(4, false);
}

TEST(Attention, GradientsMatchFiniteDifferencesCausal) {
  CheckAttentionGradients(2, true);
}

// --- padding mask ---

TEST(Attention, ValidLengthIsRangeChecked) {
  Attention attn(2);
  EXPECT_EQ(attn.valid_length(), kSeq);
  EXPECT_NO_THROW(attn.set_valid_length(1));
  EXPECT_NO_THROW(attn.set_valid_length(kSeq));
  EXPECT_THROW(attn.set_valid_length(0), std::out_of_range);
  EXPECT_THROW(attn.set_valid_length(kSeq + 1), std::out_of_range);
}

TEST(Attention, ValidLengthZeroesAttentionToPadding) {
  auto gen = zf_test::rng();
  Attention attn(2);
  attn.randomize(gen);
  attn.set_valid_length(2);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen);
  attn.forward(x);

  for (std::size_t h = 0; h < attn.heads(); ++h) {
    const auto& a = attn.attention_weights(h);
    for (std::size_t i = 0; i < kSeq; ++i) {
      float sum = 0.0f;
      for (std::size_t j = 0; j < kSeq; ++j) {
        if (j >= 2) {
          EXPECT_FLOAT_EQ(a(i, j), 0.0f) << "row " << i << " attended to pad";
        }
        sum += a(i, j);
      }
      if (i < 2) {
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "valid row " << i;
      } else {
        // A padded query is not computed at all, so its row stays empty.
        // Normalization is only meaningful for a row that exists; what has to
        // hold of the padding is that it cannot reach a valid row, which
        // PaddingCannotInfluenceValidPositions checks directly.
        EXPECT_FLOAT_EQ(sum, 0.0f) << "padded row " << i << " was computed";
      }
    }
  }
}

TEST(Attention, PaddingCannotInfluenceValidPositions) {
  auto gen = zf_test::rng();
  Attention attn(2);
  attn.randomize(gen);
  attn.set_valid_length(2);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen);
  const auto base = attn.forward(x);

  // Scribble over every padded row; the real rows must not move.
  zf::TensorF<kSeq, kModel> perturbed = x;
  for (std::size_t r = 2; r < kSeq; ++r) {
    for (std::size_t d = 0; d < kModel; ++d) {
      perturbed(r, d) += 5.0f;
    }
  }
  const auto after = attn.forward(perturbed);
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t d = 0; d < kModel; ++d) {
      EXPECT_NEAR(base(r, d), after(r, d), 1e-5f)
          << "valid row " << r << " moved";
    }
  }
}

TEST(Attention, GradientsMatchFiniteDifferencesWithPadding) {
  auto gen = zf_test::rng();
  Attention attn(2);
  attn.randomize(gen);
  attn.set_valid_length(3);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen, 0.7);
  zf::TensorF<kSeq, kModel> w;
  randomize_tensor(w, gen, 0.7);

  const auto loss = [&] { return zf::dot(attn.forward(x), w); };

  attn.zero_grad();
  attn.forward(x);
  const auto dx = attn.backward(w);

  expect_grads_match(slots_of(attn.query_proj().weight()),
                     values_of(attn.query_proj().weight_grad()), loss,
                     "masked wq");
  expect_grads_match(slots_of(attn.value_proj().weight()),
                     values_of(attn.value_proj().weight_grad()), loss,
                     "masked wv");
  expect_grads_match(slots_of(x), values_of(dx), loss, "masked input");
}

TEST(Transformer, ValidLengthReachesEveryBlock) {
  Stack model(3, 2);
  model.set_valid_length(2);
  EXPECT_EQ(model.valid_length(), 2u);
  for (std::size_t i = 0; i < model.depth(); ++i) {
    EXPECT_EQ(model.block(i).valid_length(), 2u);
  }
}

TEST(Transformer, PaddingCannotInfluenceValidPositionsThroughTheStack) {
  auto gen = zf_test::rng();
  Stack model(2, 2);
  model.randomize(gen);
  model.set_valid_length(2);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen, 0.5);
  const auto base = model.forward(x);

  zf::TensorF<kSeq, kModel> perturbed = x;
  for (std::size_t r = 2; r < kSeq; ++r) {
    for (std::size_t d = 0; d < kModel; ++d) {
      perturbed(r, d) += 5.0f;
    }
  }
  const auto after = model.forward(perturbed);
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t d = 0; d < kModel; ++d) {
      EXPECT_NEAR(base(r, d), after(r, d), 1e-4f);
    }
  }
}

TEST(TransformerBlock, ForwardPreservesShapeAndIsResidual) {
  auto gen = zf_test::rng();
  Block block(2);
  // With zeroed sublayer weights the residual path is the identity, up to the
  // layer norms, which only scale/shift. All weights start at zero, so both
  // sublayers output their (zero) bias and the block returns x unchanged.
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen);
  const auto y = block.forward(x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    EXPECT_NEAR(y[i], x[i], 1e-5f);
  }
}

TEST(TransformerBlock, GradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  Block block(2);
  block.randomize(gen);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen, 0.7);
  zf::TensorF<kSeq, kModel> w;
  randomize_tensor(w, gen, 0.7);

  const auto loss = [&] { return zf::dot(block.forward(x), w); };

  block.zero_grad();
  block.forward(x);
  const auto dx = block.backward(w);

  expect_grads_match(slots_of(block.attention().query_proj().weight()),
                     values_of(block.attention().query_proj().weight_grad()),
                     loss, "block wq");
  expect_grads_match(slots_of(block.attention().output_proj().weight()),
                     values_of(block.attention().output_proj().weight_grad()),
                     loss, "block wo");
  expect_grads_match(slots_of(block.feed_forward().input_layer().weight()),
                     values_of(block.feed_forward().input_layer().weight_grad()),
                     loss, "block ff1");
  expect_grads_match(slots_of(block.feed_forward().output_layer().bias()),
                     values_of(block.feed_forward().output_layer().bias_grad()),
                     loss, "block ff2 bias");
  expect_grads_match(slots_of(block.attention_norm().gamma()),
                     values_of(block.attention_norm().gamma_grad()), loss,
                     "block ln1 gamma");
  expect_grads_match(slots_of(block.feed_forward_norm().beta()),
                     values_of(block.feed_forward_norm().beta_grad()), loss,
                     "block ln2 beta");
  expect_grads_match(slots_of(x), values_of(dx), loss, "block input");
}

TEST(Transformer, DepthIsRuntime) {
  Stack shallow(1, 2);
  Stack deep(3, 2);
  static_assert(std::is_same_v<decltype(shallow), decltype(deep)>);
  EXPECT_EQ(shallow.depth(), 1u);
  EXPECT_EQ(deep.depth(), 3u);
  EXPECT_EQ(deep.parameter_count(),
            3 * Block::parameter_count + 2 * kModel);
  EXPECT_THROW((Stack(0, 2)), std::invalid_argument);
  EXPECT_THROW((Stack(2, 3)), std::invalid_argument);
}

TEST(Transformer, GradientsMatchFiniteDifferencesThroughTheStack) {
  auto gen = zf_test::rng();
  Stack model(2, 2);
  model.randomize(gen);
  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen, 0.7);
  zf::TensorF<kSeq, kModel> w;
  randomize_tensor(w, gen, 0.7);

  const auto loss = [&] { return zf::dot(model.forward(x), w); };

  model.zero_grad();
  model.forward(x);
  const auto dx = model.backward(w);

  // A gradient reaching block 0's weights proves the chain crosses both blocks
  // and the final norm.
  expect_grads_match(slots_of(model.block(0).attention().value_proj().weight()),
                     values_of(
                         model.block(0).attention().value_proj().weight_grad()),
                     loss, "stack block0 wv");
  expect_grads_match(slots_of(model.block(1).feed_forward_norm().gamma()),
                     values_of(model.block(1).feed_forward_norm().gamma_grad()),
                     loss, "stack block1 ln2 gamma");
  expect_grads_match(slots_of(model.final_norm().gamma()),
                     values_of(model.final_norm().gamma_grad()), loss,
                     "stack final norm gamma");
  expect_grads_match(slots_of(x), values_of(dx), loss, "stack input");
}

TEST(Transformer, SgdReducesMseOnAFixedTarget) {
  auto gen = zf_test::rng();
  Stack model(2, 2);
  model.randomize(gen);
  zf::MseLoss<float, kSeq, kModel> loss;

  zf::TensorF<kSeq, kModel> x;
  randomize_tensor(x, gen, 0.5);
  zf::TensorF<kSeq, kModel> target;
  randomize_tensor(target, gen, 0.5);

  const float before = loss.forward(model.forward(x), target);
  for (int step = 0; step < 100; ++step) {
    model.zero_grad();
    const auto pred = model.forward(x);
    loss.forward(pred, target);
    model.backward(loss.backward());
    model.apply_sgd(0.02f);
  }
  const float after = loss.forward(model.forward(x), target);
  EXPECT_LT(after, before * 0.5f);
}

}  // namespace
