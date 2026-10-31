#include "loss.h"

namespace zf {

template class MseLoss<float, 4>;
template class MseLoss<float, 8>;
template class MseLoss<float, 4, 8>;
template class SoftmaxCrossEntropyLoss<float, 4>;
template class SoftmaxCrossEntropyLoss<float, 10>;
template class InfoNceLoss<float, 4>;
template class InfoNceLoss<float, 8>;
template class MseLoss<bf16, 4>;
template class SoftmaxCrossEntropyLoss<bf16, 4>;
template class InfoNceLoss<bf16, 4>;

}  // namespace zf
