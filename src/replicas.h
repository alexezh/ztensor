#pragma once

// Data-parallel batches: a batch's examples split across threads.
//
// Replicas<Model>::run() splits a batch of n examples into at most `most`
// slices of whole examples. Slice 0 runs on the model itself; the others on
// helper copies of it, made on first use. Each slice runs forward and
// backward on its own copy -- its own activations, its own gradients -- so
// the slices share nothing while they run. Then, with training, two passes
// over the parameters, each split by tensor over the pool:
//
//   reduce   the model's gradient += the helpers', in slice order; the
//            helpers' gradients zeroed; each tensor's squared norm
//   update   w -= rate(norm²) · g, the model's gradients zeroed, and the
//            new weights copied into every helper
//
// The step size is the caller's: `rate` gets the summed gradient's squared
// norm (for clipping) and returns the factor to step by.
//
// A helper starts the next step already holding the model's weights. If
// they were changed some other way in between (a load, another trainer), a
// spot check of one weight per tensor catches it and the helpers are copied
// whole.
//
// The slices depend only on n and `most`, never on the number of threads,
// so the result is the same on one thread as on twelve (ZF_THREADS=1 runs
// the slices one after another on the caller).
//
// Slicing costs work: a pass has a fixed cost (one matrix product per
// projection, whatever its rows), so k thin slices take longer on one
// thread than one thick pass. It pays when the threads would otherwise
// idle.
//
// A model is any type with visit_layers(f) calling f on each layer; a layer
// is a Linear, LayerNorm or Embedding, or anything with its own
// visit_layers(). Each layer has zero_grad().

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensor.h"
#include "thread_pool.h"

namespace zf {

// Every parameter tensor of `layer`, in order: a Linear's weight and bias, a
// LayerNorm's gamma and beta, an Embedding's weight, or those of the layers
// it visits. A layer with its own visit_tensors() lists its own.
template <typename Layer, typename Visit>
void visit_params(Layer& layer, Visit&& visit) {
  if constexpr (requires { layer.visit_tensors(visit); }) {
    layer.visit_tensors(visit);
  } else if constexpr (requires { layer.visit_layers([](auto&) {}); }) {
    layer.visit_layers([&](auto& inner) { visit_params(inner, visit); });
  } else if constexpr (requires { layer.bias(); }) {
    visit(layer.weight());
    visit(layer.bias());
  } else if constexpr (requires { layer.gamma(); }) {
    visit(layer.gamma());
    visit(layer.beta());
  } else {
    visit(layer.weight());
  }
}

// Every gradient tensor of `layer`, in visit_params' order.
template <typename Layer, typename Visit>
void visit_grads(Layer& layer, Visit&& visit) {
  if constexpr (requires { layer.visit_layers([](auto&) {}); }) {
    layer.visit_layers([&](auto& inner) { visit_grads(inner, visit); });
  } else if constexpr (requires { layer.bias_grad(); }) {
    visit(layer.weight_grad());
    visit(layer.bias_grad());
  } else if constexpr (requires { layer.gamma_grad(); }) {
    visit(layer.gamma_grad());
    visit(layer.beta_grad());
  } else {
    visit(layer.weight_grad());
  }
}

template <typename E>
struct Block {
  E* data;
  std::size_t size;
};

// A model's parameter tensors, or its gradient tensors, as blocks of E.
template <typename E, typename Model>
std::vector<Block<E>> param_blocks(Model& model) {
  std::vector<Block<E>> out;
  model.visit_layers([&](auto& layer) {
    visit_params(layer, [&](auto& t) {
      static_assert(std::is_same_v<std::remove_reference_t<decltype(*t.data())>, E>);
      out.push_back({t.data(), t.size()});
    });
  });
  return out;
}

template <typename E, typename Model>
std::vector<Block<E>> grad_blocks(Model& model) {
  std::vector<Block<E>> out;
  model.visit_layers([&](auto& layer) {
    visit_grads(layer, [&](auto& t) {
      static_assert(std::is_same_v<std::remove_reference_t<decltype(*t.data())>, E>);
      out.push_back({t.data(), t.size()});
    });
  });
  return out;
}

// A model's helper copies. Held by the model; a copy of the model starts
// with none of its own.
template <typename Model>
class Replicas {
 public:
  Replicas() = default;
  Replicas(const Replicas&) {}
  Replicas& operator=(const Replicas&) {
    helpers_.clear();
    return *this;
  }

  // f(model, b0, b1, stats) for each slice [b0, b1) of a batch of n
  // examples, at most `most` slices, slice 0 on `master`. f runs the slice
  // forward and, with `train`, backward, adding into that model's
  // gradients. With `train`, `master` then takes one step on the sum:
  // w -= rate(norm²) g, rate returning the weights' type. Each slice's
  // stats are added into *stats, in slice order.
  template <typename Stats, typename F, typename Rate>
  void run(Model& master, std::size_t n, std::size_t most, bool train, Stats* stats, F&& f,
           Rate&& rate) {
    using T = std::invoke_result_t<Rate&, double>;
    static_assert(std::is_same_v<T, accumulator_t<T>>,
                  "the update pass assumes the gradients are of the weights' type");
    if (n == 0) return;
    if (most == 0) most = 1;
    const std::size_t grain = (n + most - 1) / most;
    const std::size_t slices = chunk_count(n, grain);
    while (helpers_.size() + 1 < slices) helpers_.push_back(std::make_unique<Model>(master));
    const Params<T> p = params<T>(master, slices);
    // One slice runs everything on the caller.
    ThreadPool* const on = slices > 1 ? pool() : nullptr;
    if (stale(p)) copy_weights(p, on);
    std::vector<Stats> part(slices);
    if (train) master.visit_layers([](auto& layer) { layer.zero_grad(); });
    parallel_for(pool(), n, grain, [&](std::size_t c, std::size_t b0, std::size_t b1) {
      f(c == 0 ? master : *helpers_[c - 1], b0, b1, &part[c]);
    });
    if (train) step(p, rate, on);
    for (const Stats& s : part) *stats += s;
  }

 private:
  static ThreadPool* pool() { return &ThreadPool::global(); }

  // The model's tensors and those of the helpers in use, tensor by tensor.
  template <typename T>
  struct Params {
    std::vector<Block<T>> w, g;                 // the model's
    std::vector<std::vector<Block<T>>> hw, hg;  // each helper's
  };

  template <typename T>
  Params<T> params(Model& master, std::size_t slices) {
    Params<T> p{param_blocks<T>(master), grad_blocks<T>(master), {}, {}};
    for (std::size_t k = 0; k + 1 < slices; ++k) {
      p.hw.push_back(param_blocks<T>(*helpers_[k]));
      p.hg.push_back(grad_blocks<T>(*helpers_[k]));
    }
    return p;
  }

  // Whether some helper's weights differ from the model's: one weight per
  // tensor, which any whole-model change (a load, an SGD step) moves.
  template <typename T>
  static bool stale(const Params<T>& p) {
    for (const auto& hw : p.hw) {
      for (std::size_t i = 0; i < p.w.size(); ++i) {
        const std::size_t j = p.w[i].size / 2;
        if (std::memcmp(&hw[i].data[j], &p.w[i].data[j], sizeof(T)) != 0) return true;
      }
    }
    return false;
  }

  template <typename T>
  static void copy_weights(const Params<T>& p, ThreadPool* on) {
    parallel_for(on, p.w.size(), 1, [&](std::size_t, std::size_t i0, std::size_t i1) {
      for (std::size_t i = i0; i < i1; ++i) {
        for (const auto& hw : p.hw) std::memcpy(hw[i].data, p.w[i].data, p.w[i].size * sizeof(T));
      }
    });
  }

  template <typename T, typename Rate>
  static void step(const Params<T>& p, Rate&& rate_of, ThreadPool* on) {
    const std::size_t tensors = p.g.size();
    // reduce
    std::vector<double> norm2(tensors, 0.0);
    parallel_for(on, tensors, 1, [&](std::size_t, std::size_t i0, std::size_t i1) {
      for (std::size_t i = i0; i < i1; ++i) {
        T* g = p.g[i].data;
        const std::size_t size = p.g[i].size;
        for (const auto& hg : p.hg) {
          T* h = hg[i].data;
          for (std::size_t j = 0; j < size; ++j) g[j] += h[j];
          std::memset(h, 0, size * sizeof(T));
        }
        // Eight independent sums: one chain through a million weights is
        // most of a step's bookkeeping.
        double lanes[8] = {};
        const std::size_t whole = size / 8 * 8;
        for (std::size_t j = 0; j < whole; j += 8) {
          for (std::size_t k = 0; k < 8; ++k) {
            const double v = static_cast<double>(g[j + k]);
            lanes[k] += v * v;
          }
        }
        for (std::size_t j = whole; j < size; ++j) {
          const double v = static_cast<double>(g[j]);
          lanes[0] += v * v;
        }
        for (double l : lanes) norm2[i] += l;
      }
    });
    double total = 0.0;
    for (double v : norm2) total += v;
    const T rate = rate_of(total);
    // update
    parallel_for(on, tensors, 1, [&](std::size_t, std::size_t i0, std::size_t i1) {
      for (std::size_t i = i0; i < i1; ++i) {
        T* w = p.w[i].data;
        T* g = p.g[i].data;
        const std::size_t size = p.w[i].size;
        for (std::size_t j = 0; j < size; ++j) w[j] -= rate * g[j];
        std::memset(g, 0, size * sizeof(T));
        for (const auto& hw : p.hw) std::memcpy(hw[i].data, w, size * sizeof(T));
      }
    });
  }

  std::vector<std::unique_ptr<Model>> helpers_;
};

}  // namespace zf
