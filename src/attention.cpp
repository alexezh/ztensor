#include "attention.h"

namespace zf {

template class MultiHeadSelfAttention<float, 4, 8>;
template class MultiHeadSelfAttention<float, 8, 16>;

}  // namespace zf
