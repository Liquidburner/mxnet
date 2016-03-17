/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cu
 * \brief
 * \author Niklas Koehler
*/

#include "./convolution3d-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution3d-inl.h"
#endif // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(Convolution3dParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNConvolution3dOp(param);
#else
  return new Convolution3dOp<gpu>(param);
#endif // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

