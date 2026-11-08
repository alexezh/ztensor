#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "rl_trainer.h"

namespace {

// Two states, two actions: the action that matches the state pays 1. A
// softmax policy per state, from logits at zero.
struct TablePolicy {
  using Obs = int;   // the state
  using Action = int;
  std::array<std::array<double, 2>, 2> logits{};
  std::size_t updates = 0;

  double p(int s, int a) const {
    const double e0 = std::exp(logits[s][0]), e1 = std::exp(logits[s][1]);
    return (a == 0 ? e0 : e1) / (e0 + e1);
  }
  void act(const std::vector<const Obs*>& seen, std::mt19937_64* rng, std::vector<Action>* out) {
    for (const Obs* s : seen) out->push_back(std::bernoulli_distribution(p(*s, 1))(*rng) ? 1 : 0);
  }
  void reinforce(const std::vector<zf::RLSample<Obs, Action>>& batch, float lr) {
    std::array<std::array<double, 2>, 2> g{};
    for (const auto& x : batch) {
      const int s = *x.seen;
      for (int a = 0; a < 2; ++a) g[s][a] += x.advantage * (p(s, a) - (a == *x.action ? 1.0 : 0.0));
    }
    for (int s = 0; s < 2; ++s) {
      for (int a = 0; a < 2; ++a) logits[s][a] -= lr * g[s][a] / static_cast<double>(batch.size());
    }
    ++updates;
  }
};

// Episodes of up to 6 ticks; the state flips each tick from a random start,
// and an episode ends early, at random, a third of the time.
struct FlipEnvs {
  std::vector<int> state, ticks;
  std::vector<bool> over;
  std::mt19937_64 rng;
  std::size_t resets = 0;

  void reset(std::size_t n, std::size_t group, std::uint64_t seed) {
    rng.seed(seed);
    state.assign(n, 0);
    ticks.assign(n, 0);
    over.assign(n, false);
    for (std::size_t i = 0; i < n; i += group) {
      const int s = static_cast<int>(rng() % 2);
      for (std::size_t k = i; k < std::min(n, i + group); ++k) state[k] = s;
    }
    ++resets;
  }
  bool done(std::size_t i) const { return over[i]; }
  void observe(const std::vector<std::size_t>& live, std::vector<int>* out) {
    for (std::size_t i : live) out->push_back(state[i]);
  }
  float step(std::size_t i, const int& a, int*) {
    const float r = a == state[i] ? 1.0f : 0.0f;
    state[i] = 1 - state[i];
    over[i] = ++ticks[i] >= 6 || rng() % 18 == 0;
    return r;
  }
};

TEST(RLTrainer, LearnsWhichActionPays) {
  TablePolicy policy;
  FlipEnvs envs;
  zf::RLTrainer<TablePolicy, FlipEnvs>::Options o;
  o.episodes = 64;
  o.group = 4;
  o.ticks = 10;
  o.batch = 32;
  zf::RLTrainer<TablePolicy, FlipEnvs> trainer(policy, envs, o);
  const auto first = trainer.round(0.5f, 1);
  EXPECT_EQ(first.episodes, 64u);
  EXPECT_EQ(first.updates, (first.steps + 31) / 32);
  EXPECT_LE(first.steps, 64u * 6);
  double last = 0.0;
  for (int k = 2; k <= 40; ++k) last = trainer.round(0.5f, k).mean_return();
  EXPECT_GT(policy.p(0, 0), 0.9);
  EXPECT_GT(policy.p(1, 1), 0.9);
  EXPECT_GT(last, first.mean_return() + 1.0);
}

// The same seed, the same round.
TEST(RLTrainer, RoundsRepeatForASeed) {
  const auto run = [] {
    TablePolicy policy;
    FlipEnvs envs;
    zf::RLTrainer<TablePolicy, FlipEnvs> trainer(policy, envs, {});
    trainer.round(0.5f, 7);
    return policy.logits;
  };
  EXPECT_EQ(run(), run());
}

}  // namespace
