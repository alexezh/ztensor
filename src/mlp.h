#pragma once

#include <cstddef>
#include <type_traits>

#include "layers.h"
#include "tensor.h"

namespace zf {

// Two-layer perceptron: InDim -> HiddenDim -> OutDim.
//
// Every dimension is a template parameter, so feeding a wrongly sized vector is
// a compile error and the parameter tensors are shape-checked. The activation
// is a runtime constructor argument because it does not change any parameter
// shape.
template <typename T, std::size_t InDim, std::size_t HiddenDim,
          std::size_t OutDim>
class Mlp {
  static_assert(std::is_floating_point_v<T>,
                "trainable models require a floating point element type");

 public:
  using value_type = T;
  using Input = Tensor<T, InDim>;
  using Hidden = Tensor<T, HiddenDim>;
  using Output = Tensor<T, OutDim>;

  static constexpr std::size_t in_dim = InDim;
  static constexpr std::size_t hidden_dim = HiddenDim;
  static constexpr std::size_t out_dim = OutDim;
  static constexpr std::size_t parameter_count =
      Linear<T, InDim, HiddenDim>::parameter_count +
      Linear<T, HiddenDim, OutDim>::parameter_count;

  explicit Mlp(Activation act = Activation::kRelu) : act_(act) {}

  Activation activation() const { return act_; }
  void set_activation(Activation act) { act_ = act; }

  template <typename Rng>
  void randomize(Rng& rng) {
    fc1_.randomize(rng);
    fc2_.randomize(rng);
  }

  Output forward(const Input& x) {
    x_ = x;
    h_pre_ = fc1_.forward(x);
    Hidden h;
    for (std::size_t i = 0; i < HiddenDim; ++i) {
      h(i) = activation_apply(act_, h_pre_(i));
    }
    h_ = h;
    return fc2_.forward(h_);
  }

  // Returns dL/dx. Must follow a matching forward().
  Input backward(const Output& d_out) {
    const Hidden dh = fc2_.backward(h_, d_out);
    Hidden dh_pre;
    for (std::size_t i = 0; i < HiddenDim; ++i) {
      dh_pre(i) = dh(i) * activation_grad(act_, h_pre_(i));
    }
    return fc1_.backward(x_, dh_pre);
  }

  void zero_grad() {
    fc1_.zero_grad();
    fc2_.zero_grad();
  }

  void apply_sgd(T lr) {
    fc1_.apply_sgd(lr);
    fc2_.apply_sgd(lr);
  }

  Linear<T, InDim, HiddenDim>& hidden_layer() { return fc1_; }
  const Linear<T, InDim, HiddenDim>& hidden_layer() const { return fc1_; }
  Linear<T, HiddenDim, OutDim>& output_layer() { return fc2_; }
  const Linear<T, HiddenDim, OutDim>& output_layer() const { return fc2_; }

 private:
  Activation act_;

  Linear<T, InDim, HiddenDim> fc1_;
  Linear<T, HiddenDim, OutDim> fc2_;

  // activations cached by forward() for use by backward()
  Input x_;
  Hidden h_pre_;
  Hidden h_;
};

}  // namespace zf
