# Reinforcement learning

`rl_trainer.h`: policy gradient in rounds. Many episodes run side by side,
then one update covers everything they did.

## A round

`RLTrainer<Model, Envs>::round(lr, seed)`:

1. **Rollout.** M episodes step in lockstep. Each tick there's one batched
   `act()` over every episode still going (a GPU-sized batch), then each
   episode's `step()` on its environment. It keeps what each episode saw, did
   and got, with no gradients and no activations.
2. **Returns.** Each step's discounted return over the rest of its episode:
   `G_t = r_t + gamma G_{t+1}`.
3. **Credit.** The advantage is `A = G - baseline`. The baseline is the mean
   return at the same tick over the episodes of its group (episodes that
   started alike), or over every episode with groups of one. Then A is
   divided by its spread over the round.
4. **Update.** `reinforce()` over every step, shuffled, in batches of
   `batch`, one optimizer step each. The policy's loss is
   `-A log pi(action | seen)`, plus whatever else it predicts and can learn
   from what happened.

A round is on-policy: one set of weights acts, then learns from what it did.
Rounds are deterministic for a seed.

```cpp
zf::RLTrainer<MyModel, MyEnvs>::Options o;
o.episodes = 256;  // M
o.group = 8;       // episodes that start alike
o.ticks = 30;      // the longest episode
o.gamma = 0.85;
o.batch = 256;     // steps per update
zf::RLTrainer<MyModel, MyEnvs> trainer(model, envs, o);
for (int r = 0; r < rounds; ++r) {
  const auto round = trainer.round(lr, seed + r);
  // round.mean_return(), round.steps, round.updates,
  // round.seconds_rollout, round.seconds_update
}
```

## The environments

One object for the round's M episodes, so observing can batch whatever else
runs per tick (another model):

```cpp
void reset(std::size_t n, std::size_t group, std::uint64_t seed);
     // n episodes; i and j start alike when i / group == j / group
bool done(std::size_t i) const;
void observe(const std::vector<std::size_t>& live, std::vector<Obs>* out);
float step(std::size_t i, const Action& a, Obs* seen);
     // the reward; may add to *seen what the tick taught (targets for the
     // policy's predictions)
```

## The policy: `RLPolicy<Model>`

The trainer knows neither the model's heads nor its losses. It reaches the
model only through `zf::RLPolicy<Model>`:

```cpp
using Obs, Action;
void act(const std::vector<const Obs*>& seen, std::mt19937_64* rng,
         std::vector<Action>* out);   // sampled, one per seen
void reinforce(const std::vector<zf::RLSample<Obs, Action>>& batch, float lr);
```

By default `RLPolicy` forwards to the model's own members of those names. A
model whose reinforcement learning is more than that (its own clip, its own
GPU step, its own stats) keeps it out of its class and specializes
`RLPolicy<Model>`. The specialization is an object over the model and can
keep state of its own; `trainer.policy()` returns it.

```cpp
template <>
struct zf::RLPolicy<Table> {
  using Obs = int;
  using Action = int;
  explicit RLPolicy(Table& t) : table(t) {}
  void act(const std::vector<const Obs*>& seen, std::mt19937_64* rng, std::vector<Action>* out);
  void reinforce(const std::vector<RLSample<Obs, Action>>& batch, float lr);
  Table& table;
  std::size_t updates = 0;
};
```

`tests/rl_trainer_test.cpp` has both forms: a tabular policy with its own
members, and the same table driven through a specialization.

## On the GPU

The rollout is forward-only and changes no weights, so a GPU model uploads
its weights once per rollout. The update's batches are where a
[whole step on the GPU](apple_gpu.md#a-whole-step-lowrescrossstep_batch) pays:
a policy's `reinforce()` can build its heads and loss (the PPO clip included)
as graph ops, and each batch is then one GPU run.
