#include "attention.h"
#include "cross_attention.h"

namespace zf {

template class MultiHeadSelfAttention<float, 4, 8>;
template class MultiHeadSelfAttention<float, 8, 16>;
template class MultiHeadSelfAttention<bf16, 4, 8>;
template class MultiHeadSelfAttention<bf16, 4, 8, 2>;
template class MultiHeadCrossAttention<bf16, 3, 5, 8>;
template class TransformerDecoder<bf16, 3, 5, 8, 16, Activation::kGelu, 2>;

}  // namespace zf
