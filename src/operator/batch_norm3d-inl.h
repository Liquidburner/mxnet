/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Florian Hendrich
*/
#ifndef MXNET_OPERATOR_BATCH_NORM3D_INL_H_
#define MXNET_OPERATOR_BATCH_NORM3D_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace batchnorm3d {
enum BatchNorm3dOpInputs {kData, kGamma, kBeta};
enum BatchNorm3dOpOutputs {kOut, kMean, kVar, kOutNoAffine};
enum BatchNorm3dOpAuxiliary {kMovingMean, kMovingVar};
enum BatchNorm3dBackResource {kTempSpace};
}  // namespace batchnorm

struct BatchNorm3dParam : public dmlc::Parameter<BatchNorm3dParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  DMLC_DECLARE_PARAMETER(BatchNorm3dParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
  }
};

template<typename xpu>
class BatchNorm3dOp : public Operator {
 public:
  explicit BatchNorm3dOp(BatchNorm3dParam param) {
    this->param_ = param;
    LOG(FATAL) << "3D Batchnorm just supported with CuDNN";

  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {   
    LOG(FATAL) << "3D Batchnorm just supported with CuDNN";
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "3D Batchnorm just supported with CuDNN";

  }

 private:
  BatchNorm3dParam param_;
};  // class BatchNorm3dOp

template<typename xpu>
Operator *CreateOp(BatchNorm3dParam param);


#if DMLC_USE_CXX11
class BatchNorm3dProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));
    if (!param_.fix_gamma) {
      out_shape->push_back(dshape);
    }
    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNorm3dProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm3d";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.fix_gamma) {
      return {out_grad[batchnorm3d::kOut],
              out_data[batchnorm3d::kMean],
              out_data[batchnorm3d::kVar],
              in_data[batchnorm3d::kData],
              in_data[batchnorm3d::kGamma],
              in_data[batchnorm3d::kBeta]
            };
    } else {
      return {out_grad[batchnorm3d::kOut],
              out_data[batchnorm3d::kOutNoAffine],
              out_data[batchnorm3d::kMean],
              out_data[batchnorm3d::kVar],
              in_data[batchnorm3d::kData],
              in_data[batchnorm3d::kGamma],
              in_data[batchnorm3d::kBeta]
            };
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[batchnorm3d::kOut], in_grad[batchnorm3d::kData]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return param_.fix_gamma ? 3 : 4;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.fix_gamma) {
      return {"output", "mean", "var"};
    } else {
      return {"output", "mean", "var", "output_no_affine"};
    }
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  BatchNorm3dParam param_;
};  // class BatchNorm3dProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_
