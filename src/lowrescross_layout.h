#pragma once

// What every LowResCross implementation shares (lowrescross.h is the
// facade): the Layout contract, a pass's inputs, the weights and where a
// batch runs. The implementations (lowrescross_cpu.h, and on the GPU
// apple/lowrescross_apple.h) hold only their activations and caches; the
// weights are the facade's, so saving, loading and the optimizer see one
// model whatever ran it.
//
// LowResCross knows nothing of what a row means. A row is a wide, sparse
// vector of kWidth floats -- most of it absent for any one row -- with a
// read mask saying which slots are present. The transformer reads a row
// through low-dimensional projections (attention in Heads x HeadDim, the
// feed-forward in Ffn) and restores what it computes into the row's state
// slots [kStateOffset, kStateOffset + kState), only where the row's write
// mask allows. The model supplies the meaning as a Layout:
//
//   struct Layout {
//     using Input = ...;      // what describes one row (the caller's token)
//     using StateGrad = ...;  // the gradient on a query row's input state,
//                             // as the caller wants it back
//     static constexpr std::size_t kWidth;        // the row
//     static constexpr std::size_t kStateOffset;  // where the state starts
//     static constexpr std::size_t kState;        // its width
//     // Learned ids: kIdSlots ids per row, slot s looked up in its own
//     // table of kIds x kIdDim and added into the row at kIdAt[s].
//     static constexpr std::size_t kIds, kIdDim, kIdSlots;
//     static constexpr std::array<std::size_t, kIdSlots> kIdAt;
//
//     template <typename T> static void put(T* row, const Input&);
//         // the row's fixed part, over a zeroed row (kWidth)
//     template <typename T> static void put_mask(T* mask, const Input&);
//         // which slots are present (kWidth), over zeros
//     template <typename T> static void put_write_mask(T* mask, const Input&);
//         // which state slots the layers may write (kState), over zeros
//     static int id(const Input&, std::size_t slot);  // -1 for none
//     template <typename T> static StateGrad state_grad(const T* d_state, const Input&);
//         // the gradient on the row's state (kState), handed back
//   };

#include <array>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "layers.h"
#include "tensor.h"

// The environment variable that turns the GPU on (1). A program may name its
// own before including this.
#ifndef ZTENSOR_LOWRES_GPU_ENV
#define ZTENSOR_LOWRES_GPU_ENV "ZTENSOR_GPU"
#endif

namespace zf {

// One pass of a batch: its query rows over its memory rows. Both must
// outlive the pass's backward().
template <typename Input>
struct LowResPass {
  const std::vector<Input>* queries = nullptr;
  const std::vector<Input>* memory = nullptr;
};

// --- where a batch runs ------------------------------------------------------
//
// A GPU run has a fixed cost of a few milliseconds, whatever its size, so
// the GPU pays only for big batches: with the GPU on, a batch of
// kLowResGpuMinPasses passes or more runs there, and anything smaller stays
// on the CPU.

enum class LowResDevice {
  kAuto,  // by batch size (lowres_device_for)
  kCpu,
  kApple,
};

// ZTENSOR_LOWRES_GPU_ENV=1 from the environment, on a build with ztensor's
// GPU target.
inline bool lowres_gpu_enabled() {
#if defined(ZTENSOR_APPLE_GPU)
  static const bool on = [] {
    const char* env = std::getenv(ZTENSOR_LOWRES_GPU_ENV);
    return env != nullptr && std::strtol(env, nullptr, 10) > 0;
  }();
  return on;
#else
  return false;
#endif
}

inline constexpr std::size_t kLowResGpuMinPasses = 64;

// The GPU's training batch: a step costs about 4 ms whatever its size.
inline constexpr std::size_t kLowResGpuBatch = 256;

inline LowResDevice lowres_device_for(std::size_t passes) {
  return lowres_gpu_enabled() && passes >= kLowResGpuMinPasses ? LowResDevice::kApple
                                                                : LowResDevice::kCpu;
}

// --- the weights -------------------------------------------------------------

// The id embeddings, then each layer. HD is the attention width (heads x
// head dim), Ffn the feed-forward's. Every map reads the whole row and the
// output maps write only the state.
template <typename T, typename Layout, std::size_t HD, std::size_t Ffn>
struct LowResCrossWeights {
  static constexpr std::size_t kWidth = Layout::kWidth;
  static constexpr std::size_t kState = Layout::kState;
  using Input = typename Layout::Input;
  using Proj = Linear<T, kWidth, HD>;
  struct Layer {
    LayerNorm<T, kWidth> n1, n2, nm, n3;
    Proj qs, ks, vs, qc, kc, vc;
    Linear<T, HD, kState> os, oc;
    Linear<T, kWidth, Ffn> f1;
    Linear<T, Ffn, kState> f2;
  };

  explicit LowResCrossWeights(std::size_t levels) : layers(levels) {}

  template <typename Rng>
  void randomize(Rng& rng) {
    for (auto& e : ids) e.randomize(rng);
    for (Layer& l : layers) {
      for (auto* p : {&l.qs, &l.ks, &l.vs, &l.qc, &l.kc, &l.vc}) p->randomize(rng);
      l.os.randomize(rng);
      l.oc.randomize(rng);
      l.f1.randomize(rng);
      l.f2.randomize(rng);
      // Start each layer near the identity: small writes into the state.
      scale_weights(l.os, T{0.1});
      scale_weights(l.oc, T{0.1});
      scale_weights(l.f2, T{0.1});
    }
  }

  // The order every implementation takes them in (and the files keep).
  template <typename Visit>
  void visit_layers(Visit&& visit) {
    for (auto& e : ids) visit(e);
    for (Layer& l : layers) {
      for (auto* n : {&l.n1, &l.n2, &l.nm, &l.n3}) visit(*n);
      for (auto* p : {&l.qs, &l.ks, &l.vs, &l.qc, &l.kc, &l.vc}) visit(*p);
      visit(l.os);
      visit(l.oc);
      visit(l.f1);
      visit(l.f2);
    }
  }

  // A whole row: its fixed part, then the learned id fields.
  void put(T* x, const Input& t) const {
    Layout::put(x, t);
    for (std::size_t s = 0; s < Layout::kIdSlots; ++s) {
      const int id = Layout::id(t, s);
      if (id < 0) continue;
      const auto e = ids[s].forward(static_cast<std::size_t>(id));
      T* at = x + Layout::kIdAt[s];
      for (std::size_t k = 0; k < Layout::kIdDim; ++k) at[k] += e(k);
    }
  }

  // The gradient on a row (kWidth) into its id embeddings.
  void identity_backward(const Input& t, const T* d) {
    Tensor<T, Layout::kIdDim> g;
    for (std::size_t s = 0; s < Layout::kIdSlots; ++s) {
      const int id = Layout::id(t, s);
      if (id < 0) continue;
      for (std::size_t k = 0; k < Layout::kIdDim; ++k) g(k) = d[Layout::kIdAt[s] + k];
      ids[s].backward(static_cast<std::size_t>(id), g);
    }
  }

  std::array<Embedding<T, Layout::kIds, Layout::kIdDim>, Layout::kIdSlots> ids;
  std::vector<Layer> layers;

 private:
  template <typename L>
  static void scale_weights(L& l, T s) {
    for (std::size_t i = 0; i < l.weight().size(); ++i) l.weight().data()[i] *= s;
  }
};

}  // namespace zf
