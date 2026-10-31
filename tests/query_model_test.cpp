#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "dot.h"
#include "grad_check.h"
#include "loss.h"
#include "query_model.h"

namespace {

using zf_test::expect_grads_match;
using zf_test::randomize_tensor;
using zf_test::slots_of;
using zf_test::values_of;

constexpr std::size_t kVocab = 32;
constexpr std::size_t kEmbed = 8;
constexpr std::size_t kHidden = 16;
constexpr std::size_t kOut = 4;

using Tower = zf::QueryModel<float, kVocab, kEmbed, kHidden, kOut>;

TEST(QueryModel, DimensionsAreCompileTime) {
  static_assert(Tower::vocab_size == kVocab);
  static_assert(Tower::embed_dim == kEmbed);
  static_assert(Tower::out_dim == kOut);
  // 32*8 embedding + (8*16+16) + (16*4+4)
  static_assert(Tower::parameter_count == 256 + 144 + 68);
}

TEST(QueryModel, OutputIsUnitNorm) {
  auto gen = zf_test::rng();
  Tower q;
  q.randomize(gen);
  const auto e = q.forward({3, 7, 11});
  static_assert(std::is_same_v<std::decay_t<decltype(e)>, zf::TensorF<kOut>>);
  EXPECT_NEAR(std::sqrt(zf::dot(e, e)), 1.0f, 1e-5f);
}

TEST(QueryModel, NormalizationCanBeDisabled) {
  auto gen = zf_test::rng();
  Tower q(zf::Activation::kRelu, /*normalize=*/false);
  q.randomize(gen);
  const auto e = q.forward({1, 2});
  EXPECT_GT(std::abs(std::sqrt(zf::dot(e, e)) - 1.0f), 1e-4f)
      << "unnormalized output should not land on the unit sphere by accident";
}

TEST(QueryModel, TokenOrderDoesNotMatterButMultiplicityDoes) {
  auto gen = zf_test::rng();
  Tower q;
  q.randomize(gen);
  const auto a = q.forward({1, 5, 9});
  const auto b = q.forward({9, 1, 5});
  for (std::size_t i = 0; i < kOut; ++i) {
    EXPECT_NEAR(a(i), b(i), 1e-6f) << "mean pooling is order invariant";
  }
  const auto c = q.forward({1, 1, 5, 9});
  bool differs = false;
  for (std::size_t i = 0; i < kOut; ++i) {
    if (std::abs(a(i) - c(i)) > 1e-5f) {
      differs = true;
    }
  }
  EXPECT_TRUE(differs) << "repeating a token reweights the mean";
}

TEST(QueryModel, PooledIsTheMeanOfEmbeddingRows) {
  Tower q(zf::Activation::kRelu, /*normalize=*/false);
  for (std::size_t i = 0; i < kEmbed; ++i) {
    q.embedding().weight()(2, i) = 2.0f;
    q.embedding().weight()(4, i) = 4.0f;
  }
  q.forward({2, 4});
  for (std::size_t i = 0; i < kEmbed; ++i) {
    EXPECT_FLOAT_EQ(q.pooled()(i), 3.0f);
  }
}

TEST(QueryModel, RejectsEmptyQueryAndBadTokens) {
  Tower q;
  EXPECT_THROW(q.forward({}), std::invalid_argument);
  EXPECT_THROW(q.forward({kVocab}), std::out_of_range);
}

TEST(QueryModel, BackwardBeforeForwardThrows) {
  Tower q;
  EXPECT_THROW(q.backward(zf::TensorF<kOut>{}), std::logic_error);
}

TEST(QueryModel, ScoreOfIdenticalNormalizedEmbeddingsIsOne) {
  auto gen = zf_test::rng();
  Tower q;
  q.randomize(gen);
  const auto e = q.forward({3, 7});
  EXPECT_NEAR(Tower::score(e, e), 1.0f, 1e-5f);
}

TEST(QueryModel, GradientsMatchFiniteDifferences) {
  auto gen = zf_test::rng();
  // GELU rather than the ReLU default: finite differences straddle ReLU's kink.
  // The embedding also gets a wider init than the 0.02 default, because mean
  // pooling a near-zero embedding leaves a near-zero vector for the L2 step,
  // and y = x/||x|| has enormous curvature there -- the analytic gradient is
  // fine, but central differences at that point are not.
  Tower q(zf::Activation::kGelu);
  q.embedding().randomize(gen, 0.5f);
  q.projection().randomize(gen);
  const std::vector<std::size_t> tokens{3, 7, 7, 20};
  zf::TensorF<kOut> w;
  randomize_tensor(w, gen);

  const auto loss = [&] { return zf::dot(q.forward(tokens), w); };

  q.zero_grad();
  q.forward(tokens);
  q.backward(w);

  expect_grads_match(slots_of(q.projection().hidden_layer().weight()),
                     values_of(q.projection().hidden_layer().weight_grad()),
                     loss, "query proj fc1");
  expect_grads_match(slots_of(q.projection().output_layer().weight()),
                     values_of(q.projection().output_layer().weight_grad()),
                     loss, "query proj fc2");
  expect_grads_match(slots_of(q.embedding().weight()),
                     values_of(q.embedding().weight_grad()), loss,
                     "query embedding");
}

TEST(QueryModel, GradientsMatchFiniteDifferencesWithoutNormalization) {
  auto gen = zf_test::rng();
  Tower q(zf::Activation::kGelu, /*normalize=*/false);
  q.embedding().randomize(gen, 0.5f);
  q.projection().randomize(gen);
  const std::vector<std::size_t> tokens{1, 2, 3};
  zf::TensorF<kOut> w;
  randomize_tensor(w, gen);

  const auto loss = [&] { return zf::dot(q.forward(tokens), w); };

  q.zero_grad();
  q.forward(tokens);
  q.backward(w);

  expect_grads_match(slots_of(q.embedding().weight()),
                     values_of(q.embedding().weight_grad()), loss,
                     "unnormalized query embedding");
}

TEST(QueryModel, UntouchedEmbeddingRowsGetNoGradient) {
  auto gen = zf_test::rng();
  Tower q;
  q.randomize(gen);
  q.zero_grad();
  q.forward({5});
  q.backward(zf::TensorF<kOut>{1, 1, 1, 1});
  for (std::size_t i = 0; i < kEmbed; ++i) {
    EXPECT_FLOAT_EQ(q.embedding().weight_grad()(6, i), 0.0f);
  }
}

// Trains a two-tower retrieval pair with InfoNCE and checks the positive
// document rises above the sampled negatives.
TEST(QueryModel, TwoTowerTrainingRanksThePositiveFirst) {
  auto gen = zf_test::rng(7);
  Tower query_tower;
  Tower doc_tower;
  query_tower.randomize(gen);
  doc_tower.randomize(gen);
  zf::InfoNceLoss<float, kOut> loss(/*temperature=*/0.1f);

  const std::vector<std::size_t> query{1, 2, 3};
  const std::vector<std::vector<std::size_t>> docs{
      {10, 11, 12},  // the positive
      {20, 21},
      {25, 26, 27},
      {30},
  };
  constexpr std::size_t kPositive = 0;

  const auto encode_docs = [&] {
    std::vector<zf::TensorF<kOut>> out;
    for (const auto& d : docs) {
      out.push_back(doc_tower.forward(d));
    }
    return out;
  };

  const float before = loss.forward(query_tower.forward(query), encode_docs(),
                                    kPositive);

  for (int step = 0; step < 300; ++step) {
    query_tower.zero_grad();
    doc_tower.zero_grad();

    const auto q_emb = query_tower.forward(query);
    const auto doc_embs = encode_docs();
    loss.forward(q_emb, doc_embs, kPositive);

    query_tower.forward(query);  // restore the query tower's cache
    query_tower.backward(loss.query_grad());
    for (std::size_t i = 0; i < docs.size(); ++i) {
      doc_tower.forward(docs[i]);  // restore this doc's cache
      doc_tower.backward(loss.candidate_grad(i));
    }

    query_tower.apply_sgd(0.1f);
    doc_tower.apply_sgd(0.1f);
  }

  const auto q_emb = query_tower.forward(query);
  const auto doc_embs = encode_docs();
  const float after = loss.forward(q_emb, doc_embs, kPositive);

  EXPECT_LT(after, before);
  EXPECT_EQ(loss.argmax(), kPositive);
  for (std::size_t i = 1; i < docs.size(); ++i) {
    EXPECT_GT(Tower::score(q_emb, doc_embs[kPositive]),
              Tower::score(q_emb, doc_embs[i]))
        << "positive should outrank negative " << i;
  }
}

}  // namespace
