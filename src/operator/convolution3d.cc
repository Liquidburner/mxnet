/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Niklas Koehler
*/

#include "./convolution3d-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(Convolution3dParam param) {
  return new Convolution3dOp<cpu>(param);
}

Operator* Convolution3dProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(Convolution3dParam);

MXNET_REGISTER_OP_PROPERTY(Convolution3d, Convolution3dProp)
.add_argument("data", "Symbol", "Input data to the Convolution3dOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(Convolution3dParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

