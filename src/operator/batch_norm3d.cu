/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cu
 * \brief
 * \author Florian Hendrich
*/

#include "./batch_norm3d-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(BatchNorm3dParam param) {
  return new BatchNorm3dOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

