/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Florian Hendrich
*/

#include "./batch_norm3d-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BatchNorm3dParam param) {
  return new BatchNorm3dOp<cpu>(param);
}

Operator *BatchNorm3dProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(BatchNorm3dParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm3d, BatchNorm3dProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(BatchNorm3dParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

