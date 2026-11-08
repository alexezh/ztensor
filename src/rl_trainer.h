#pragma once

// Reinforcement learning by policy gradient: rounds of many episodes run
// side by side, then one update over everything they did.
//
// RLTrainer<Model, Envs>::round():
//
//   rollout  M episodes step in lockstep: each tick, one batched act() over
//            every episode still going (a GPU-sized batch), then each one's
//            step() on its environment. What it saw, what it did and what
//            it got are kept; no gradients, no activations.
//   returns  each step's discounted return over the rest of its episode,
//            G_t = r_t + gamma G_{t+1}
//   credit   advantage A = G - baseline: the mean return of the episodes of
//            its group at the same tick (episodes that started alike), or of
//            every episode at that tick with groups of one; then divided by
//            the spread of A over the round
//   update   reinforce() over every step, in batches of `batch`, each one
//            optimizer step: the policy's loss is -A log pi(action | seen),
//            plus whatever it predicts and can be taught by what happened
//
// The policy samples and learns; the trainer knows neither its heads nor
// its losses. It reaches the model through RLPolicy<Model>:
//
//   using Obs, Action;
//   void act(const std::vector<const Obs*>& seen, std::mt19937_64* rng,
//            std::vector<Action>* out);         // sampled, one per seen
//   void reinforce(const std::vector<RLSample<Obs, Action>>& batch, float lr);
//
// By default RLPolicy forwards to the model's own members of those names. A
// model whose reinforcement learning is more than that (its own clip, its
// own GPU step) keeps it out of its class: it specializes RLPolicy<Model>,
// an object over the model that may keep state of its own (its stats).
//
// The environments, vectorized -- one object for the round's M episodes, so
// observing can batch whatever else runs per tick (another model):
//
//   void reset(std::size_t n, std::size_t group, std::uint64_t seed);
//            // n episodes; episodes i and j start alike when i / group ==
//            // j / group
//   bool done(std::size_t i) const;
//   void observe(const std::vector<std::size_t>& live, std::vector<Obs>* out);
//   float step(std::size_t i, const Action& a, Obs* seen);
//            // the reward; may add to *seen what the tick taught (targets
//            // for the policy's predictions)
//
// The round is on-policy: one set of weights acts, then learns from what it
// did. Rounds are deterministic for a seed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace zf {

template <typename Obs, typename Action>
struct RLSample {
  const Obs* seen = nullptr;
  const Action* action = nullptr;
  float advantage = 0.0f;
};

// The model as RLTrainer drives it: its own members.
template <typename Model>
struct RLPolicy {
  using Obs = typename Model::Obs;
  using Action = typename Model::Action;

  explicit RLPolicy(Model& model) : model_(model) {}
  void act(const std::vector<const Obs*>& seen, std::mt19937_64* rng, std::vector<Action>* out) {
    model_.act(seen, rng, out);
  }
  void reinforce(const std::vector<RLSample<Obs, Action>>& batch, float lr) { model_.reinforce(batch, lr); }

 private:
  Model& model_;
};

template <typename Model, typename Envs>
class RLTrainer {
 public:
  using Policy = RLPolicy<Model>;
  using Obs = typename Policy::Obs;
  using Action = typename Policy::Action;
  using Sample = RLSample<Obs, Action>;

  struct Options {
    std::size_t episodes = 256;  // M, per round
    std::size_t group = 8;       // episodes that start alike
    std::size_t ticks = 30;      // the longest episode
    double gamma = 0.85;         // discount per tick
    std::size_t batch = 256;     // steps per update
  };

  struct Round {
    std::size_t episodes = 0, steps = 0, updates = 0;
    double reward = 0.0;  // summed over every step
    double seconds_rollout = 0.0, seconds_update = 0.0;
    double mean_return() const { return episodes == 0 ? 0.0 : reward / static_cast<double>(episodes); }
  };

  RLTrainer(Model& model, Envs& envs, Options options) : policy_(model), envs_(envs), o_(options) {}

  Policy& policy() { return policy_; }

  Round round(float lr, std::uint64_t seed) {
    Round r;
    r.episodes = o_.episodes;
    std::mt19937_64 rng(seed);
    const auto t0 = std::chrono::steady_clock::now();
    rollout(&rng, &r);
    const auto t1 = std::chrono::steady_clock::now();
    const std::vector<Sample> samples = credit();
    update(samples, lr, &rng, &r);
    r.steps = samples.size();
    r.seconds_rollout = std::chrono::duration<double>(t1 - t0).count();
    r.seconds_update = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    return r;
  }

 private:
  struct Episode {
    std::vector<Obs> seen;
    std::vector<Action> actions;
    std::vector<float> rewards;
  };

  void rollout(std::mt19937_64* rng, Round* r) {
    const std::size_t m = o_.episodes;
    envs_.reset(m, std::max<std::size_t>(1, o_.group), (*rng)());
    episodes_.assign(m, {});
    std::vector<std::size_t> live;
    std::vector<Obs> seen;
    std::vector<const Obs*> ptrs;
    std::vector<Action> actions;
    for (std::size_t t = 0; t < o_.ticks; ++t) {
      live.clear();
      for (std::size_t i = 0; i < m; ++i) {
        if (!envs_.done(i)) live.push_back(i);
      }
      if (live.empty()) break;
      seen.clear();
      envs_.observe(live, &seen);
      ptrs.clear();
      for (const Obs& s : seen) ptrs.push_back(&s);
      actions.clear();
      policy_.act(ptrs, rng, &actions);
      for (std::size_t k = 0; k < live.size(); ++k) {
        Episode& ep = episodes_[live[k]];
        ep.seen.push_back(std::move(seen[k]));
        ep.actions.push_back(std::move(actions[k]));
        const float reward = envs_.step(live[k], ep.actions.back(), &ep.seen.back());
        ep.rewards.push_back(reward);
        r->reward += reward;
      }
    }
  }

  // Every step with its advantage.
  std::vector<Sample> credit() const {
    const std::size_t m = episodes_.size();
    std::vector<std::vector<double>> g(m);
    for (std::size_t i = 0; i < m; ++i) {
      const auto& rw = episodes_[i].rewards;
      g[i].assign(rw.size(), 0.0);
      double run = 0.0;
      for (std::size_t t = rw.size(); t-- > 0;) {
        run = rw[t] + o_.gamma * run;
        g[i][t] = run;
      }
    }
    // The baseline: the mean return at the same tick, over the group's
    // episodes still going then (or every episode's, with groups of one).
    const std::size_t group = std::max<std::size_t>(1, o_.group);
    const std::size_t pool = group == 1 ? m : group;
    std::vector<std::vector<double>> a(m);
    for (std::size_t i = 0; i < m; ++i) a[i].resize(g[i].size());
    for (std::size_t g0 = 0; g0 < m; g0 += pool) {
      const std::size_t g1 = std::min(m, g0 + pool);
      for (std::size_t t = 0; t < o_.ticks; ++t) {
        double sum = 0.0;
        std::size_t n = 0;
        for (std::size_t i = g0; i < g1; ++i) {
          if (t < g[i].size()) {
            sum += g[i][t];
            ++n;
          }
        }
        if (n == 0) break;
        const double mean = sum / static_cast<double>(n);
        for (std::size_t i = g0; i < g1; ++i) {
          if (t < g[i].size()) a[i][t] = n > 1 ? g[i][t] - mean : 0.0;
        }
      }
    }
    double s2 = 0.0;
    std::size_t n = 0;
    for (const auto& ai : a) {
      for (double v : ai) {
        s2 += v * v;
        ++n;
      }
    }
    const double spread = n == 0 ? 0.0 : std::sqrt(s2 / static_cast<double>(n));
    const double scale = spread > 1e-8 ? 1.0 / spread : 0.0;
    std::vector<Sample> out;
    out.reserve(n);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t t = 0; t < a[i].size(); ++t) {
        out.push_back({&episodes_[i].seen[t], &episodes_[i].actions[t], static_cast<float>(a[i][t] * scale)});
      }
    }
    return out;
  }

  void update(const std::vector<Sample>& samples, float lr, std::mt19937_64* rng, Round* r) {
    std::vector<std::size_t> order(samples.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::shuffle(order.begin(), order.end(), *rng);
    std::vector<Sample> batch;
    for (std::size_t b0 = 0; b0 < order.size(); b0 += o_.batch) {
      const std::size_t b1 = std::min(order.size(), b0 + o_.batch);
      batch.clear();
      for (std::size_t k = b0; k < b1; ++k) batch.push_back(samples[order[k]]);
      policy_.reinforce(batch, lr);
      ++r->updates;
    }
  }

  Policy policy_;
  Envs& envs_;
  Options o_;
  std::vector<Episode> episodes_;
};

}  // namespace zf
