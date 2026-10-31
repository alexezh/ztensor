#include "add.h"

namespace zf {

template TensorF<3> add(const TensorF<3>&, const TensorF<3>&);
template TensorF<3> sub(const TensorF<3>&, const TensorF<3>&);
template TensorF<3>& add_into(TensorF<3>&, const TensorF<3>&);
template TensorF<3>& add_scaled_into(TensorF<3>&, const TensorF<3>&, float);

template TensorF<4, 8> add(const TensorF<4, 8>&, const TensorF<4, 8>&);
template TensorF<4, 8>& add_into(TensorF<4, 8>&, const TensorF<4, 8>&);
template TensorF<4, 8>& add_scaled_into(TensorF<4, 8>&, const TensorF<4, 8>&,
                                        float);

template TensorB<4, 8> add(const TensorB<4, 8>&, const TensorB<4, 8>&);
template TensorB<4, 8>& add_into(TensorB<4, 8>&, const TensorB<4, 8>&);
template TensorB<4, 8>& add_scaled_into(TensorB<4, 8>&, const TensorB<4, 8>&,
                                        bf16);

}  // namespace zf
