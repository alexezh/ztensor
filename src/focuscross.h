#pragma once

// FocusCross: LowResCross with a workspace that thinks too.
//
// The rows are LowResCross's (lowrescross_layout.h: wide, sparse, with
// state slots the layers write), and so are the queries' layers. What
// changes is the memory, here the workspace: it is no longer fixed through
// the stack. Every level first lets the workspace rows attend to each other,
// gated by how relevant the queries found each row at the level before,
// then runs the queries over the workspace as it now stands:
//
//   workspace  w_j += r_j · mask · O_w(attn_j(Q_w ŵ, K_w ŵ + log(r + eps), V_w ŵ))
//   self       h   += mask · O_s(attn(Q_s x̂, K_s x̂, V_s x̂))        among the queries
//   cross      h   += mask · O_c(attn(Q_c x̂', K_c ŵ', V_c ŵ'))     into the workspace
//              r_j  = 1 - exp(-s_j), s_j = the attention row j got,
//                     summed over the pass's query rows, averaged over heads
//   ffn        h   += mask · W_2 relu(W_1 x̂'')
//
// The relevance r is continuous, so the loss reaches it: a row matters when
// the queries look at it, and a row that matters both takes part in the
// workspace's attention (as a key, through log(r + eps)) and is changed by
// it (the gate r_j). A row nobody looked at stays as it was and is barely
// seen by the others. At the first level every row counts (r = 1); each
// level's cross attention sets the relevance the next level's workspace
// attention works with, so what is relevant can change as the stack goes:
// a relation found at one level can make another row matter at the next.
//
// The workspace rows evolve in their state slots, where their write mask
// allows (Layout::put_write_mask): an internal working memory, apart from
// the caller's rows. memory_rows() are the rows after the last level, what
// a pointer (LowResPointer) scores against.
//
// The workspace rows depend on the pass's queries, so there is no memories'
// cache (LowResCross::set_memories).
//
// Where it runs, as LowResCross: on the CPU (focuscross_cpu.h) or on the GPU
// (apple/focuscross_apple.h), picked per batch by set_device() (kAuto: the
// big batches go to the GPU when it is on); step_batch() runs a whole step,
// the model's heads included, in one GPU run.

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "focuscross_cpu.h"
#include "layers.h"
#include "lowrescross_layout.h"
#include "tensor.h"
#include "thread_pool.h"

#if defined(ZTENSOR_APPLE_GPU)
#include "apple/focuscross_apple.h"
#endif

namespace zf {

template <typename T, typename Layout, std::size_t Q, std::size_t M, std::size_t Heads = 4,
          std::size_t HeadDim = 8, std::size_t Ffn = 32>
class FocusCross {
 public:
  static constexpr std::size_t HD = Heads * HeadDim;
  static constexpr std::size_t kQ = Q, kM = M, kHeads = Heads, kHeadDim = HeadDim, kFfn = Ffn;
  static constexpr std::size_t kStream = Layout::kWidth;
  static constexpr std::size_t kHidden = Layout::kState;
  using Input = typename Layout::Input;
  using StateGrad = typename Layout::StateGrad;
  using Pass = LowResPass<Input>;
  using Device = LowResDevice;
  using Weights = FocusCrossWeights<T, Layout, HD, Ffn>;
  using Hidden = Tensor<T, Q, kHidden>;
  using Readout = Tensor<T, kHidden>;

  explicit FocusCross(std::size_t levels) : w_(levels) {}

  std::size_t levels() const { return w_.layers.size(); }

  // Where batches run; kAuto (the default) is lowres_device_for(their size).
  void set_device(Device d) { device_ = d; }
  // The device the last batch ran on.
  Device ran_on() const { return on_; }

  // The CPU's thread pool; nullptr: everything on the caller.
  void set_pool(ThreadPool* pool) { cpu_.set_pool(pool); }

  template <typename Rng>
  void randomize(Rng& rng) {
    w_.randomize(rng);
    ++w_.version;
  }

  // Every pass of `batch` at once: each pass's queries over its workspace.
  void forward_batch(const std::vector<Pass>& batch) {
    pick(batch.size());
    each([&](auto& e) { e.forward_batch(w_, batch); });
  }

  // One pass (on the CPU). Returns the readouts (the raw final state of
  // each query row).
  const Hidden& forward(const std::vector<Input>& queries, const std::vector<Input>& workspace) {
    on_ = Device::kCpu;
    cpu_.forward_batch(w_, {Pass{&queries, &workspace}});
    readout0_ = readout_of(0);
    return readout0_;
  }

  // --- the batch just run ---

  std::size_t batch_size() const {
    return each([](const auto& e) { return e.batch_size(); });
  }
  std::size_t query_size(std::size_t b) const {
    return each([&](const auto& e) { return e.query_size(b); });
  }

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
  // Query row i's raw final state, slots [at, at + N) of the state.
  template <std::size_t N>
  std::array<T, N> state(std::size_t b, std::size_t i, std::size_t at) const {
    std::array<T, N> s{};
    const T* f = hidden_row(b, i) + at;
    for (std::size_t k = 0; k < N; ++k) s[k] = f[k];
    return s;
  }

  // Pass b's workspace rows after the last level: memory_size rows of
  // kStream, contiguous.
  std::size_t memory_size(std::size_t b = 0) const {
    return each([&](const auto& e) { return e.memory_size(b); });
  }
  const T* memory_rows(std::size_t b = 0) const {
    return each([&](const auto& e) { return e.memory_rows(b); });
  }
  // The gradient on those rows, from a pointer head (added into); consumed
  // by backward().
  T* memory_grad(std::size_t b = 0) {
    return each([&](auto& e) { return e.memory_grad(b); });
  }

  // Pass b's relevance of each workspace row (memory_size values in [0, 1)),
  // as the cross attention of `level` left it; by default the last level's.
  const T* relevance(std::size_t b = 0) const { return relevance(b, levels() - 1); }
  const T* relevance(std::size_t b, std::size_t level) const {
    return each([&](const auto& e) { return e.relevance(b, level); });
  }

  // `d_readout[b]` on pass b's readouts. Adds into the parameter gradients;
  // returns dL/d(input state) per pass, per query row.
  std::vector<std::vector<StateGrad>> backward_batch(const std::vector<Hidden>& d_readout) {
    return each([&](auto& e) { return e.backward_batch(w_, d_readout); });
  }
  std::vector<StateGrad> backward(const Hidden& d_readout) {
    return std::move(backward_batch({d_readout})[0]);
  }

  // --- a whole step on the GPU ---
  //
  // As LowResCross::step_batch: forward, the model's heads (their concept is
  // in lowrescross.h) and every gradient in one GPU run. The heads build on
  // the readouts and on the workspace rows after the last level.

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
    throw std::logic_error("FocusCross: step_batch() runs on the GPU only");
  }

  // The layers, to read or change (LowResCross's order, then each level's
  // workspace attention).
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
  using Cpu = FocusCrossCpu<T, Layout, Q, M, Heads, HeadDim, Ffn>;
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
    if (device_ == Device::kApple && !kGpu) throw std::logic_error("FocusCross: no GPU for this model");
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
  using Gpu = FocusCrossApple<T, Layout, Q, M, Heads, HeadDim, Ffn>;
  LowResGpuSlot<Gpu> gpu_;
#endif
  Hidden readout0_{};
};

}  // namespace zf
