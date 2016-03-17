/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm.cc
 * \brief
 * \author Florian Hendrich
*/

#include "./cudnn_batch_norm3d-inl.h"
namespace mxnet {
namespace op {
#if CUDNN_MAJOR >= 4
template<>
Operator *CreateOp<cpu>(CuDNNBatchNorm3dParam param) {
  LOG(FATAL) << "CuDNNBatchNormOp is only available for gpu.";
  return NULL;
}

Operator *CuDNNBatchNorm3dProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CuDNNBatchNorm3dParam);

MXNET_REGISTER_OP_PROPERTY(CuDNNBatchNorm3d, CuDNNBatchNorm3dProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(CuDNNBatchNorm3dParam::__FIELDS__());
#endif  // CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
