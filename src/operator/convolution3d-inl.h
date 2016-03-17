/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution-inl.h
 * \brief
 * \author Niklas Koehler
*/
#ifndef MXNET_OPERATOR_CONVOLUTION3D_INL_H_
#define MXNET_OPERATOR_CONVOLUTION3D_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"


namespace mxnet {
namespace op {

namespace conv {
enum Convolution3dOpInputs {kData, kWeight, kBias};
enum Convolution3dOpOutputs {kOut};
enum Convolution3dOpResource {kTempSpace};
}

struct Convolution3dParam : public dmlc::Parameter<Convolution3dParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(Convolution3dParam) {
    int shape[] = {1, 1, 1};
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (z, y, x)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 3))
    .describe("convolution stride: (z, y, x)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 3))
    .describe("convolution dilate: (z,y, x)");
    shape[0] = shape[1] = shape[2] = 0;
    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 3))
    .describe("pad for convolution: (z,y, x)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of groups partition. "
              "This option is not supported by CuDNN, you can use SliceChannel to num_group,"
              "apply convolution and concat instead to achieve the same need.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(128, 4096)
    .describe("Tmp workspace for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
  }
};

template<typename xpu>
class Convolution3dOp : public Operator {
 public:
  explicit Convolution3dOp(Convolution3dParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
    
    LOG(FATAL) << "3D Convolution just supported with CuDNN";
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
 
       LOG(FATAL) << "3D Convolution just supported with CuDNN";
    
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
   
      LOG(FATAL) << "3D Convolution just supported with CuDNN";
    
  }

 private:
    Convolution3dParam param_;
};  // class ConvolutionOp

template<typename xpu>
Operator* CreateOp(Convolution3dParam param);

#if DMLC_USE_CXX11
class Convolution3dProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[conv::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 5) \
        << "Input data should be 4D in batch-num_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape,
                       conv::kWeight,
                       Shape5(param_.num_filter, dshape[1], param_.kernel[0],
			      param_.kernel[1], param_.kernel[2]));
   if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_z = static_cast<index_t>(param_.kernel[0]);    
    const index_t ksize_y = static_cast<index_t>(param_.kernel[1]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[2]);

    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    CHECK_GE(param_.dilate.Size(), 0) \
        << "incorrect dilate size: " << param_.dilate;
    CHECK(ksize_x <= dshape[4] && ksize_y <= dshape[3] && ksize_z <= dshape[2])
        << "kernel size exceed input";
    (*out_shape)[conv::kOut][1] = param_.num_filter;

    (*out_shape)[conv::kOut][2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] == 1 ? ksize_z : ksize_z * param_.dilate[0] - 1)) / param_.stride[0] + 1;

    (*out_shape)[conv::kOut][3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] == 1 ? ksize_y : ksize_y * param_.dilate[1] - 1)) / param_.stride[1] + 1;


    (*out_shape)[conv::kOut][4] = (dshape[4] + 2 * param_.pad[2] -
        (param_.dilate[2] == 1 ? ksize_x : ksize_x * param_.dilate[2] - 1)) / param_.stride[2] + 1;
     return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new Convolution3dProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Convolution3d";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[conv::kOut], in_data[conv::kData], in_data[conv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  Convolution3dParam param_;
};  // class Convolution3dProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_
