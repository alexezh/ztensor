#include "query_model.h"

namespace zf {

template class QueryModel<float, 32, 8, 16, 4>;
template class QueryModel<float, 64, 16, 32, 8>;

}  // namespace zf
