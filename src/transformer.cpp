#include "transformer.h"

namespace zf {

template class FeedForward<float, 4, 8, 16>;
template class TransformerBlock<float, 4, 8, 16>;
template class Transformer<float, 4, 8, 16>;
template class FeedForward<float, 4, 8, 16, Activation::kRelu>;
template class Transformer<float, 4, 8, 16, Activation::kRelu>;
template class Transformer<float, 4, 8, 16, Activation::kGelu, 2>;

}  // namespace zf
