/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling3d.cc
 * \brief
 * \author Philipp Eulenberg
*/
#include "./pooling3d-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(Pooling3dParam param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return new Pooling3dOp<cpu, mshadow::red::maximum>(param);
    case pool_enum::kAvgPooling:
      return new Pooling3dOp<cpu, mshadow::red::sum>(param);
    case pool_enum::kSumPooling:
      return new Pooling3dOp<cpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

Operator* Pooling3dProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(Pooling3dParam);

MXNET_REGISTER_OP_PROPERTY(Pooling3d, Pooling3dProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(Pooling3dParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

