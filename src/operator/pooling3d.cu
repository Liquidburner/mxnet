/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cu
 * \brief
 * \author Philipp Eulenberg
*/

#include "./pooling3d-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_pooling3d-inl.h"
#endif // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(Pooling3dParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNPooling3dOp(param);
#else
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return new Pooling3dOp<gpu, mshadow::red::maximum>(param);
    case pool_enum::kAvgPooling:
      return new Pooling3dOp<gpu, mshadow::red::sum>(param);
    case pool_enum::kSumPooling:
      return new Pooling3dOp<gpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
#endif  // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

