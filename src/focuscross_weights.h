#pragma once

// What every FocusCross implementation shares (focuscross.h is the facade):
// the weights, and the relevance's epsilon. The rows, passes and devices are
// LowResCross's (lowrescross_layout.h).

#include <cstddef>
#include <vector>

#include "layers.h"
#include "lowrescross_layout.h"

namespace zf {

// What a key's relevance r adds to its scores in the workspace attention:
// log(r + kFocusEps). A row nobody looked at is a key at about 1/100th of
// the weight of one that matters; the gradient through the log is at most
// 1 / kFocusEps.
inline constexpr double kFocusEps = 0.01;

// The weights: LowResCross's (the ids, and per level the query self
// attention, the cross attention into the workspace and the feed-forward),
// and per level the workspace's self attention.
template <typename T, typename Layout, std::size_t HD, std::size_t Ffn>
struct FocusCrossWeights : LowResCrossWeights<T, Layout, HD, Ffn> {
  using Base = LowResCrossWeights<T, Layout, HD, Ffn>;
  using Proj = typename Base::Proj;
  struct Focus {
    LayerNorm<T, Layout::kWidth> nw;
    Proj qw, kw, vw;
    Linear<T, HD, Layout::kState> ow;
  };

  explicit FocusCrossWeights(std::size_t levels) : Base(levels), focus(levels) {}

  template <typename Rng>
  void randomize(Rng& rng) {
    Base::randomize(rng);
    for (Focus& f : focus) {
      for (auto* p : {&f.qw, &f.kw, &f.vw}) p->randomize(rng);
      f.ow.randomize(rng);
      // Near the identity, as the other output maps.
      for (std::size_t i = 0; i < f.ow.weight().size(); ++i) f.ow.weight().data()[i] *= T{0.1};
    }
  }

  // LowResCross's order, then each level's workspace attention.
  template <typename Visit>
  void visit_layers(Visit&& visit) {
    Base::visit_layers(visit);
    for (Focus& f : focus) {
      visit(f.nw);
      for (auto* p : {&f.qw, &f.kw, &f.vw}) visit(*p);
      visit(f.ow);
    }
  }

  std::vector<Focus> focus;
};

}  // namespace zf
