#pragma once

// LowResCross: a cross transformer over wide, sparse rows, computed at low
// width.
//
// Every query row and every memory row is one wide vector (Layout::kWidth,
// lowrescross_layout.h), mostly absent for any one row: a read mask says
// which slots are present. Part of the row is state (Layout::kState slots
// at Layout::kStateOffset) -- the residual stream the layers write; the
// rest is fixed through the stack. The layers read the whole row, through a
// norm over its present slots, project it down to the width they compute
// in, and restore the result into the state, only where the row's write
// mask allows:
//
//   self   h += mask · O_s(attn(Q_s x̂, K_s x̂, V_s x̂))     among the queries
//   cross  h += mask · O_c(attn(Q_c x̂', K_c m̂, V_c m̂))   into the memory
//   ffn    h += mask · W_2 relu(W_1 x̂'')                    within the row
//
// Attention works in Heads x HeadDim and the feed-forward in Ffn: small
// enough to keep the transformer cheap however wide the row. The memory is
// fixed for the pass (decoder-only). What a row means -- which slots, which
// ids, what its state gradient looks like to the caller -- is the Layout's.
//
// This class is a facade: it holds the weights, and an implementation runs
// them -- lowrescross_cpu.h, or apple/lowrescross_apple.h on the GPU
// (lowres_device_for says which: by default the GPU takes the big batches
// when it is on). The weights stay here, so saving, loading and the
// optimizer see one model whatever ran it; an implementation keeps only its
// activations and its cache, and adds into the weights' gradients.
//
// Batches: forward_batch() runs many passes at once -- different examples,
// each with its own queries and memory; passes never see each other. A pass
// is indexed by its place in the batch, `b`; forward() is a batch of one
// (always on the CPU).
//
// The memories' cache: many passes over the same few memories read memories
// the implementation keeps. A memory row's cross K and V depend on that row
// alone, so each row goes through the memory side once, when it is added,
// and rows can be added and removed without touching the others. Query rows
// are not cached: every pass's queries attend to each other both ways. The
// memories' gradients add up over every backward_batch() until
// backward_memories() takes them back through once; the memories cannot
// change in between, nor can the weights. When the weights change (a step)
// the cache's K and V are made again from them before the next pass reads
// them. Any forward_batch() ends the cache.
//
// Weights and gradients on the GPU: the implementation keeps its own copy
// of the weights and takes them again only when they have changed (a
// version the facade moves on randomize(), apply_sgd() and visit_layers()),
// so a model that runs many batches in a step uploads its weights once.
// Its gradients add up on the GPU over a sequence of runs -- a batch's
// backward, or every pass over the memories' cache until
// backward_memories() -- and come back into the weights' gradients once at
// its end.
//
// Readouts are read off the final state (readout()), unnormalized: a
// readout row has few present slots, and a norm over its mostly-absent
// state would divide by a near-zero spread. backward() takes the gradient
// on that state and returns, per query row, the gradient on its input state
// (Layout::state_grad), so a model can train through state it wrote in an
// earlier pass.
//
// LowResPointer: a bilinear score of a query row's readout against every
// memory row, (A · readout) · (B · memory row). Order-free.

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "layers.h"
#include "lowrescross_cpu.h"
#include "lowrescross_layout.h"
#include "tensor.h"
#include "thread_pool.h"

#if defined(ZTENSOR_APPLE_GPU)
#include "apple/lowrescross_apple.h"
#endif

namespace zf {

template <typename T, typename Layout, std::size_t Q, std::size_t M, std::size_t Heads = 4,
          std::size_t HeadDim = 8, std::size_t Ffn = 32>
class LowResCross {
 public:
  static constexpr std::size_t HD = Heads * HeadDim;
  static constexpr std::size_t kQ = Q, kM = M, kHeads = Heads, kHeadDim = HeadDim, kFfn = Ffn;
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  using Input = typename Layout::Input;
  using StateGrad = typename Layout::StateGrad;
  using Pass = LowResPass<Input>;
  using Device = LowResDevice;
  using Weights = LowResCrossWeights<T, Layout, HD, Ffn>;
  using Stream = Tensor<T, kStream>;
  using Hidden = Tensor<T, Q, kHidden>;
  using Readout = Tensor<T, kHidden>;

  explicit LowResCross(std::size_t levels) : w_(levels) {}

  std::size_t levels() const { return w_.layers.size(); }

  // Where batches run; kAuto (the default) is lowres_device_for(their size).
  void set_device(Device d) { device_ = d; }
  // The device the last batch or memory ran on.
  Device ran_on() const { return on_; }

  // The CPU's thread pool; nullptr: everything on the caller.
  void set_pool(ThreadPool* pool) { cpu_.set_pool(pool); }

  template <typename Rng>
  void randomize(Rng& rng) {
    w_.randomize(rng);
    ++w_.version;
  }

  // Every pass of `batch` at once.
  void forward_batch(const std::vector<Pass>& batch) {
    pick(batch.size());
    each([&](auto& e) { e.forward_batch(w_, batch); });
  }

  // One pass: the query tokens over the memory tokens. Returns the readouts
  // (the raw final hidden state of each query row).
  const Hidden& forward(const std::vector<Input>& queries, const std::vector<Input>& memory) {
    on_ = Device::kCpu;
    cpu_.forward_batch(w_, {Pass{&queries, &memory}});
    readout0_ = readout_of(0);
    return readout0_;
  }

  // --- the memories' cache ---

  // Replaces the memories with `memories`; pass b of forward_shared() reads
  // the one memory_of[b] names. On the device for that many memories.
  void set_memories(const std::vector<const std::vector<Input>*>& memories) {
    pick(memories.size());
    each([&](auto& e) { e.set_memories(w_, memories); });
  }
  // One more memory; returns its index.
  std::size_t add_memory(const std::vector<Input>& rows) {
    return each([&](auto& e) { return e.add_memory(w_, rows); });
  }
  // `rows` (indices into memory j) out of memory j; the rest keep their
  // place in order.
  void remove_memory_rows(std::size_t j, const std::vector<std::size_t>& rows) {
    each([&](auto& e) { e.remove_memory_rows(j, rows); });
  }
  std::size_t memory_count() const {
    return each([](const auto& e) { return e.memory_count(); });
  }
  void forward_shared(const std::vector<const std::vector<Input>*>& queries,
                      const std::vector<std::size_t>& memory_of) {
    each([&](auto& e) { e.forward_shared(w_, queries, memory_of); });
  }
  // The memories' gradients, summed over every pass since they last
  // changed, back through their K, V and norms once. Adds into the
  // parameter gradients.
  void backward_memories() {
    each([&](auto& e) { e.backward_memories(w_); });
  }

  // --- the batch just run ---

  std::size_t batch_size() const {
    return each([](const auto& e) { return e.batch_size(); });
  }
  std::size_t query_size(std::size_t b) const {
    return each([&](const auto& e) { return e.query_size(b); });
  }

  // Pass b's readouts; readout() is pass 0's of the last forward().
  Hidden readout_of(std::size_t b) const {
    Hidden h{};
    for (std::size_t i = 0; i < query_size(b); ++i) {
      const T* f = hidden_row(b, i);
      for (std::size_t k = 0; k < kHidden; ++k) h(i, k) = f[k];
    }
    return h;
  }
  const Hidden& readout() const { return readout0_; }
  Readout readout(std::size_t b, std::size_t i) const {
    Readout v;
    const T* f = hidden_row(b, i);
    for (std::size_t k = 0; k < kHidden; ++k) v(k) = f[k];
    return v;
  }

  // Query row i's raw final state, slots [at, at + N) of the state (in
  // pass b).
  template <std::size_t N>
  std::array<T, N> state(std::size_t b, std::size_t i, std::size_t at) const {
    std::array<T, N> s{};
    const T* f = hidden_row(b, i) + at;
    for (std::size_t k = 0; k < N; ++k) s[k] = f[k];
    return s;
  }

  // Pass b's memory rows as the pointer and the stack see them: memory_size
  // rows of kStream, contiguous.
  std::size_t memory_size(std::size_t b = 0) const {
    return each([&](const auto& e) { return e.memory_size(b); });
  }
  const T* memory_rows(std::size_t b = 0) const {
    return each([&](const auto& e) { return e.memory_rows(b); });
  }
  // The gradient on those rows, from a pointer head (added into); consumed
  // by backward() (or, for the memories' cache, backward_memories()).
  T* memory_grad(std::size_t b = 0) {
    return each([&](auto& e) { return e.memory_grad(b); });
  }

  // `d_readout[b]` on pass b's readouts. Adds into the parameter gradients;
  // returns dL/d(input hidden) per pass, per query token.
  std::vector<std::vector<StateGrad>> backward_batch(const std::vector<Hidden>& d_readout) {
    return each([&](auto& e) { return e.backward_batch(w_, d_readout); });
  }

  // A batch of one. `d_state` (optional) is more of the same, for a model
  // that keeps the two apart.
  std::vector<StateGrad> backward(const Hidden& d_readout, const Hidden* d_state = nullptr) {
    std::vector<Hidden> d{d_readout};
    if (d_state != nullptr) {
      for (std::size_t i = 0; i < d[0].size(); ++i) d[0].data()[i] += d_state->data()[i];
    }
    return std::move(backward_batch(d)[0]);
  }

  // --- a whole step on the GPU ---
  //
  // step_batch() runs forward, the model's heads and their loss, and every
  // gradient in one GPU run: no readouts come back, no backward_batch(). The
  // heads are the model's, built from graph ops (apple/lowrescross_graph.h):
  //
  //   template <typename Visit> void visit_layers(Visit&&);  // the heads' layers
  //   struct Nodes;  // what build() keeps of the graph
  //   apple::Node build(LowResHeadsGraph&, Nodes*);
  //        // the heads on hg.hidden and hg.rows, taking their parameters
  //        // with hg.params.next(), their inputs with hg.in.add() and their
  //        // results with hg.out.add(); returns the loss, summed over passes
  //   void fill(const Nodes&, LowResGraphInputs&);
  //        // this batch's inputs, over zeros (a padded pass stays zero and
  //        // must add nothing)
  //   void read(const Nodes&, const LowResGraphOutputs&);  // after the run
  //
  // The stack's gradients add into its weights', the heads' into theirs.
  // On the CPU, a model does the same with forward_batch(), its heads and
  // backward_batch().

  // Whether step_batch() takes a batch of n passes: where forward_batch()
  // would run it, the GPU.
  bool steps_on_gpu(std::size_t n) const {
    return kGpu && (device_ != Device::kAuto ? device_ == Device::kApple : lowres_device_for(n) == Device::kApple);
  }
  template <typename HeadSet>
  void step_batch(const std::vector<Pass>& batch, HeadSet& heads) {
#if defined(ZTENSOR_APPLE_GPU)
    if constexpr (kGpu) {
      if (steps_on_gpu(batch.size())) {
        on_ = Device::kApple;
        gpu_.get().step_batch(w_, batch, heads);
        return;
      }
    }
#endif
    (void)batch;
    (void)heads;
    throw std::logic_error("LowResCross: step_batch() runs on the GPU only");
  }

  // The layers, to read or change: the weights count as changed (an
  // optimizer, a load, a copy into a replica), so the implementation takes
  // them afresh before its next run. A pointer kept from one visit must not
  // write into the weights after the next run.
  template <typename Visit>
  void visit_layers(Visit&& visit) {
    ++w_.version;
    w_.visit_layers(visit);
  }

  void zero_grad() {
    w_.visit_layers([](auto& l) { l.zero_grad(); });
  }
  void apply_sgd(T lr) {
    visit_layers([lr](auto& l) { l.apply_sgd(lr); });
  }

 private:
  using Cpu = LowResCrossCpu<T, Layout, Q, M, Heads, HeadDim, Ffn>;
#if defined(ZTENSOR_APPLE_GPU)
  static constexpr bool kGpu = std::is_same_v<T, float>;
#else
  static constexpr bool kGpu = false;
#endif

  const T* hidden_row(std::size_t b, std::size_t i) const {
    return each([&](const auto& e) { return e.hidden_row(b, i); });
  }

  // The device for a batch of n. The GPU runs float models only.
  void pick(std::size_t n) {
    if (device_ == Device::kApple && !kGpu) throw std::logic_error("LowResCross: no GPU for this model");
    on_ = !kGpu ? Device::kCpu : device_ != Device::kAuto ? device_ : lowres_device_for(n);
  }

  // f on the implementation the last batch ran on.
  template <typename F>
  decltype(auto) each(F&& f) {
#if defined(ZTENSOR_APPLE_GPU)
    if constexpr (kGpu) {
      if (on_ == Device::kApple) return f(gpu_.get());
    }
#endif
    return f(cpu_);
  }
  template <typename F>
  decltype(auto) each(F&& f) const {
#if defined(ZTENSOR_APPLE_GPU)
    if constexpr (kGpu) {
      if (on_ == Device::kApple) return f(gpu_.get());
    }
#endif
    return f(cpu_);
  }

  Weights w_;
  Device device_ = Device::kAuto;
  Device on_ = Device::kCpu;
  Cpu cpu_;
#if defined(ZTENSOR_APPLE_GPU)
  using Gpu = LowResCrossApple<T, Layout, Q, M, Heads, HeadDim, Ffn>;
  // Made on first use; a copy of the model starts without one (the GPU's
  // buffers are not the copy's).
  struct GpuSlot {
    GpuSlot() = default;
    GpuSlot(const GpuSlot&) {}
    GpuSlot& operator=(const GpuSlot&) {
      p.reset();
      return *this;
    }
    Gpu& get() const {
      if (p == nullptr) p = std::make_shared<Gpu>();
      return *p;
    }
    mutable std::shared_ptr<Gpu> p;
  };
  GpuSlot gpu_;
#endif
  Hidden readout0_{};
};

// A pointer from a query row's readout into a pass's memory rows:
// logit_j = (A · readout) · (B · memory_j).
template <typename T, typename Layout, std::size_t M, std::size_t K = 16>
class LowResPointer {
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;

 public:
  template <typename Rng>
  void randomize(Rng& rng) {
    a_.randomize(rng);
    b_.randomize(rng);
  }

  template <typename Cross>
  Tensor<T, M> forward(const Cross& cross, const Tensor<T, kHidden>& readout,
                           std::size_t b = 0) {
    qa_ = a_.forward(readout);
    Tensor<T, M> logits{};
    keys_ = Tensor<T, M, K>{};
    const std::size_t n = cross.memory_size(b);
    if (n > 0) b_.forward_into(cross.memory_rows(b), keys_.data(), n);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < K; ++k) logits(j) += qa_(k) * keys_(j, k);
    }
    return logits;
  }

  // Right after forward() on the same pass. Returns dL/d(readout); the
  // memory rows' gradients go to `cross`.
  template <typename Cross>
  Tensor<T, kHidden> backward(Cross& cross, const Tensor<T, kHidden>& readout,
                                  const Tensor<T, M>& d_logits, std::size_t b = 0) {
    Tensor<T, K> d_qa{};
    Tensor<T, M, K> d_keys{};
    const std::size_t n = cross.memory_size(b);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < K; ++k) {
        d_qa(k) += d_logits(j) * keys_(j, k);
        d_keys(j, k) = d_logits(j) * qa_(k);
      }
    }
    if (n > 0) b_.backward_into(cross.memory_rows(b), d_keys.data(), cross.memory_grad(b), n);
    return a_.backward(readout, d_qa);
  }

  template <typename Visit>
  void visit_layers(Visit&& visit) {
    visit(a_);
    visit(b_);
  }
  void zero_grad() {
    a_.zero_grad();
    b_.zero_grad();
  }
  void apply_sgd(T lr) {
    a_.apply_sgd(lr);
    b_.apply_sgd(lr);
  }

 private:
  Linear<T, kHidden, K> a_;
  Linear<T, kStream, K> b_;
  Tensor<T, K> qa_;
  Tensor<T, M, K> keys_;
};

}  // namespace zf
