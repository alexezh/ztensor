#include "mlp.h"

namespace zf {

template class Mlp<float, 4, 8, 2>;
template class Mlp<float, 8, 16, 8>;
template class Mlp<float, 8, 16, 4>;
template class Mlp<bf16, 4, 8, 2>;

}  // namespace zf
