/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_convolution-inl.h
 * \brief
 * \author Niklas Koehler
*/
#ifndef MXNET_OPERATOR_CUDNN_CONVOLUTION3D_INL_H_
#define MXNET_OPERATOR_CUDNN_CONVOLUTION3D_INL_H_

#include <algorithm>
#include <vector>
#include "./convolution3d-inl.h"

namespace mxnet {
namespace op {
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1
class CuDNNConvolution3dOp : public Operator {
 public:
  explicit CuDNNConvolution3dOp(Convolution3dParam param) {
    this->param_ = param;
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
    init_cudnn_ = false;
    // TODO(xxx): fp16
    dtype_ = CUDNN_DATA_FLOAT;
  }

  ~CuDNNConvolution3dOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyFilterDescriptor(filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyConvolutionDescriptor(conv_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 5> data = in_data[conv::kData].get<gpu, 5, real_t>(s);
    Tensor<gpu, 5> wmat = in_data[conv::kWeight].get<gpu, 5, real_t>(s);
    Tensor<gpu, 5> out = out_data[conv::kOut].get<gpu, 5, real_t>(s);
    
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(wmat.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }
    Tensor<gpu, 1> workspace = ctx.requested[conv::kTempSpace].get_space<gpu>(
                                 mshadow::Shape1(forward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      float alpha = 1.0f;
      float beta = 0.0f;
            
      CHECK_EQ(cudnnConvolutionForward(s->dnn_handle_,
                                       &alpha,
                                       in_desc_,
                                       data.dptr_ + data_offset_ * g,
                                       filter_desc_,
                                       wmat.dptr_ + weight_offset_ * g,
                                       conv_desc_,
                                       algo_,
                                       workspace.dptr_,
                                       forward_workspace_byte_,
                                       &beta,
                                       out_desc_,
                                       out.dptr_ + out_offset_ * g), CUDNN_STATUS_SUCCESS);
      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1> bias = in_data[conv::kBias].get<gpu, 1, real_t>(s);
#if CUDNN_MAJOR == 4
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta,
                                out_desc_,
                                out.dptr_ + out_offset_ * g), CUDNN_STATUS_SUCCESS);
#endif
#if CUDNN_MAJOR == 3
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                CUDNN_ADD_SAME_C,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta,
                                out_desc_,
                                out.dptr_ + out_offset_ * g), CUDNN_STATUS_SUCCESS);
#endif
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    
    
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    // TODO(bing): think about how to support add to
    CHECK_EQ(req[conv::kWeight], kWriteTo);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 5> grad = out_grad[conv::kOut].get<gpu, 5, real_t>(s);
    Tensor<gpu, 5> wmat = in_data[conv::kWeight].get<gpu, 5, real_t>(s);
    Tensor<gpu, 5> gwmat = in_grad[conv::kWeight].get<gpu, 5, real_t>(s);
    Tensor<gpu, 5> data = in_data[conv::kData].get<gpu, 5, real_t>(s);
    Tensor<gpu, 5> gdata = in_grad[conv::kData].get<gpu, 5, real_t>(s);
    Tensor<gpu, 1> workspace = ctx.requested[conv::kTempSpace].get_space<gpu>(
                                 mshadow::Shape1(backward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      float alpha = 1.0f;
      float beta = 0.0f;
      if (!param_.no_bias) {
        Tensor<gpu, 1> gbias = in_grad[conv::kBias].get<gpu, 1, real_t>(s);
        CHECK_EQ(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              out_desc_,
                                              grad.dptr_ + out_offset_ * g,
                                              &beta,
                                              bias_desc_,
                                              gbias.dptr_ + bias_offset_ * g),
					      CUDNN_STATUS_SUCCESS);
      }
      CHECK_EQ(cudnnConvolutionBackwardFilter_v3(s->dnn_handle_,
               &alpha,
               in_desc_,
               data.dptr_ + data_offset_ * g,
               out_desc_,
               grad.dptr_ + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               filter_desc_,
               gwmat.dptr_ + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat.dptr_ + weight_offset_ * g,
               out_desc_,
               grad.dptr_ + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata.dptr_ + data_offset_ * g), CUDNN_STATUS_SUCCESS);
    }
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(real_t));
      size_t back_size = 0;
      size_t back_size_w = 0;
      
      
      Tensor<gpu, 5> data = in_data[conv::kData].get<gpu, 5, real_t>(s);
      Tensor<gpu, 5> out = out_data[conv::kOut].get<gpu, 5, real_t>(s);
      
      
      CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);
      
      
      int filterDimA[5] = {int(param_.num_filter) / int(param_.num_group), int(data.shape_[1]) / int(param_.num_group), int(param_.kernel[0]), int(param_.kernel[1]), int(param_.kernel[2])};
      CHECK_EQ(cudnnSetFilterNdDescriptor(filter_desc_,
					  dtype_,
                                          5, // nbdims
					  filterDimA
					  ), CUDNN_STATUS_SUCCESS);
      
      int conv_padA[3] = {int(param_.pad[0]),int(param_.pad[1]),int(param_.pad[2])};
      int conv_strideA[3] = {int(param_.stride[0]),int(param_.stride[1]),int(param_.stride[2])};
      int conv_upscaleA[3] = {1,1,1};
      
      CHECK_EQ(cudnnSetConvolutionNdDescriptor_v3(	conv_desc_,
						3, // dimension of the convolution
						conv_padA,
						conv_strideA,
						conv_upscaleA,
						CUDNN_CROSS_CORRELATION,
						dtype_
						), CUDNN_STATUS_SUCCESS);
      
      
      int in_tensorDimA[5] = {int(data.shape_[0]),int(data.shape_[1])/ int(param_.num_group),int(data.shape_[2]),int(data.shape_[3]),int(data.shape_[4])};
      int in_tensorStrideA[5] = {int(data.shape_[1])*int(data.shape_[2])*int(data.shape_[3])*int(data.shape_[4]),
				int(data.shape_[2])*int(data.shape_[3])*int(data.shape_[4]),
				int(data.shape_[3])*int(data.shape_[4]),
				int(data.shape_[4]),
				1};
      CHECK_EQ(cudnnSetTensorNdDescriptor(in_desc_,
                                          dtype_,
					  5, // nbdims
                                          in_tensorDimA,
                                          in_tensorStrideA
					  ), CUDNN_STATUS_SUCCESS);

      int out_tensorDimA[5] = 		{int(out.shape_[0]),int(out.shape_[1]) / int(param_.num_group),int(out.shape_[2]),int(out.shape_[3]),int(out.shape_[4])};
      
      int out_tensorStrideA[5] = 	{int(out.shape_[1])*int(out.shape_[2])*int(out.shape_[3])*int(out.shape_[4]),
					int(out.shape_[2])*int(out.shape_[3])*int(out.shape_[4]),
					int(out.shape_[3])*int(out.shape_[4]),
					int(out.shape_[4]),
					1};

      CHECK_EQ(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
					  5, // nbdims
					  out_tensorDimA,
					  out_tensorStrideA
					  ), CUDNN_STATUS_SUCCESS);

      if (!param_.no_bias) {
        Tensor<gpu, 1> bias = in_data[conv::kBias].get<gpu, 1, real_t>(s);
        bias_offset_ = bias.shape_[0] / param_.num_group;
        
        /* toDo: Check bias tensor dimensions
	CHECK_EQ(cudnnSetTensor4dDescriptor(bias_desc_,
                                            CUDNN_TENSOR_NCHW,
                                            dtype_,
                                            1,
                                            bias.shape_[0] / param_.num_group,
                                            1,
                                            1), CUDNN_STATUS_SUCCESS);
	*/
	
	int bias_tensorDimA[5] = {1, int(bias.shape_[0]) /  int(param_.num_group),1, 1, 1};
        int bias_tensorStrideA[5] = {	bias_tensorDimA[1]*bias_tensorDimA[2]*bias_tensorDimA[3]*bias_tensorDimA[4],
					bias_tensorDimA[2]*bias_tensorDimA[3]*bias_tensorDimA[4],
					bias_tensorDimA[3]*bias_tensorDimA[4],
					bias_tensorDimA[4],
					1};
        CHECK_EQ(cudnnSetTensorNdDescriptor(bias_desc_,
                                            dtype_,
					    5, // nbdims
					    bias_tensorDimA,
					    bias_tensorStrideA
					), CUDNN_STATUS_SUCCESS);
	
	
      }
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
               workspace_byte,
               &algo_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
               workspace_byte,
               &back_algo_w_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
               workspace_byte,
               &back_algo_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               back_algo_,
               &back_size), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               back_algo_w_,
               &back_size_w), CUDNN_STATUS_SUCCESS);
      backward_workspace_byte_ = std::max(back_size, back_size_w);
      CHECK_EQ(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               algo_,
               &forward_workspace_byte_), CUDNN_STATUS_SUCCESS);
      forward_workspace_ = forward_workspace_byte_ / sizeof(real_t) + 1;
      backward_workspace_ = backward_workspace_byte_ / sizeof(real_t) + 1;
      
      
      data_offset_ = data.shape_[1] / param_.num_group * data.shape_[2] * data.shape_[3];
      out_offset_ = out.shape_[1] /param_.num_group * out.shape_[2] * out.shape_[3];
      weight_offset_ = param_.num_filter / param_.num_group * data.shape_[1] / param_.num_group
                       * param_.kernel[0] * param_.kernel[1];
      
		       
    }
  }

  bool init_cudnn_;
  size_t forward_workspace_;
  size_t backward_workspace_;
  size_t forward_workspace_byte_;
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  Convolution3dParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_
