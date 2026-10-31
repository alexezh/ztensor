#include "layers.h"

namespace zf {

template class Linear<float, 8, 16>;
template class Linear<float, 16, 8>;
template class Linear<float, 8, 8>;
template class LayerNorm<float, 8>;
template class Embedding<float, 32, 8>;

template float activation_apply(Activation, float);
template float activation_grad(Activation, float);
template void softmax_inplace(float*, std::size_t);

template class Linear<bf16, 8, 16>;
template class Linear<bf16, 6, 5>;
template class LayerNorm<bf16, 8>;
template class Embedding<bf16, 32, 8>;
template void softmax_inplace(bf16*, std::size_t);

}  // namespace zf
