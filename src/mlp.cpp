#include "mlp.h"

namespace zf {

template class Mlp<float, 4, 8, 2>;
template class Mlp<float, 8, 16, 8>;
template class Mlp<float, 8, 16, 4>;

}  // namespace zf
