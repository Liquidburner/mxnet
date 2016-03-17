/*!
 *  Copyright (c) 2015 by Florian Hendrich
 * \file iter_imglst.cc
 * \brief define a CSV Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_prefetcher.h"
#include "./iter_batchloader.h"
#include <opencv2/opencv.hpp>
#include "./image_augmenter_multichannel.h"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace mxnet {
namespace io {
// CSV parameters
struct IMGListIterParam : public dmlc::Parameter<IMGListIterParam> {
  /*! \brief path to the csv file */
  std::string img_lst_path;
  /*! \brief input shape */
  TShape data_shape;
  /*! \brief path to label csv file */
  bool is_train;
  /*! \brief path to label csv file */
  TShape label_shape;
  // declare parameters
  index_t seed;
  // number of random time-slices
  index_t rand_time_slices_num;
  // number of random saxes to draw
  index_t num_rand_saxes;
  DMLC_DECLARE_PARAMETER(IMGListIterParam) {
    DMLC_DECLARE_FIELD(img_lst_path)
        .describe("Dataset Param: Image-list path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("Dataset Param: Shape of the data.");
    DMLC_DECLARE_FIELD(is_train).set_default(false)
        .describe("Dataset Param: label is available");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("Dataset Param: Shape of the label.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Dataset Param: Seed for list shuffle.");
    DMLC_DECLARE_FIELD(rand_time_slices_num).set_default(0)
        .describe("Dataset Param: Number of random time-slices.");
    DMLC_DECLARE_FIELD(num_rand_saxes).set_default(0)
        .describe("Dataset Param: Number of random saxes to use.");
  }
};

struct ListEntry{
  index_t id;
  std::vector<std::string> data_paths;
  std::vector<real_t> label;
};

class IMGListIter: public IIterator<DataInst> {
 public:
  IMGListIter() {
    out_.data.resize(2);
  }
  virtual ~IMGListIter() {
    mshadow::FreeSpace(&src_tensor_3d);
    if (param_.data_shape.ndim() == 4) {
      mshadow::FreeSpace(&src_tensor_4d);
    }
  }

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    
    /* Create a new ImageAugmenter */
    augmenter_ = std::unique_ptr<ImageAugmenter_Multichannel>(new ImageAugmenter_Multichannel());
    augmenter_->Init(kwargs);
    /* Create random number generator */
    prnd_ = std::unique_ptr<common::RANDOM_ENGINE>(new common::RANDOM_ENGINE(kRandMagic));
    /* Read img list */
    this->LoadImglist_();
    /* Create a dummy label if needed */
    if (!param_.is_train) {
      dummy_label.set_pad(false);
      dummy_label.Resize(mshadow::Shape1(1));
      dummy_label = 0.0f;
    }
    src_tensor_3d = mshadow::NewTensor<cpu, real_t>(mshadow::Shape3(param_.data_shape[0], param_.data_shape[1], param_.data_shape[2]), 0.0);
    
    // Check if input size matches num_channels*30
    CHECK(param_.num_rand_saxes == param_.data_shape[0]/30 || param_.num_rand_saxes == 0) 
	<< "rand_time_slices_num != data_shape[1] " << param_.rand_time_slices_num << " != " << param_.data_shape[1];
    
    if (param_.data_shape.ndim() == 4) { 
      CHECK(param_.rand_time_slices_num == param_.data_shape[1] || param_.rand_time_slices_num == 0) 
	<< "rand_time_slices_num != data_shape[1] " << param_.rand_time_slices_num << " != " << param_.data_shape[1];
      src_tensor_4d = mshadow::NewTensor<cpu, real_t>(mshadow::Shape4(param_.data_shape[0], param_.data_shape[1], param_.data_shape[2], param_.data_shape[3]), 0.0);
      block_mean_ = mshadow::NewTensor<cpu, real_t>(mshadow::Shape2(param_.data_shape[2], param_.data_shape[3]), 0.0);
    }
    else {
      CHECK(param_.rand_time_slices_num == param_.data_shape[0] || param_.rand_time_slices_num == 0) 
	<< "rand_time_slices_num != data_shape[0] " << param_.rand_time_slices_num << " != " << param_.data_shape[0];
      block_mean_ = mshadow::NewTensor<cpu, real_t>(mshadow::Shape2(param_.data_shape[1], param_.data_shape[2]), 0.0);
    }

    //this->BeforeFirst();
  }

  virtual void BeforeFirst() {    
    /* snuffle imglst */    
    if(param_.is_train) std::shuffle(data_.begin(), data_.end(), std::default_random_engine(param_.seed));
    /* reset instance counter */
    inst_counter_ = 0;
  }

  virtual bool Next() {
    if ( inst_counter_ >= data_.size() ) return false;
    this->ReadInstance_(inst_counter_++);
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

 private:
  
  inline void LoadImglist_(void) {
    std::ifstream dataStream (param_.img_lst_path.c_str(), std::ios::in);
    CHECK(dataStream)
      << "Data list file not found: " << param_.img_lst_path;
    
    index_t path_idx = param_.is_train ? 2 : 1;      
    while (dataStream) {
      // Read on line from the list
      ListEntry tmp;
      std::string s;
      if (!std::getline( dataStream, s )) break;
      std::istringstream ss( s );
      index_t line_count = 0;
      while (ss) {
	
	// Read one entry of the line
	std::string s;
	if (!getline( ss, s, '\t' )) break;
	// read id from list
	if ( line_count % 3 == 0 ) tmp.id = atoi(s.c_str());
	else if ( line_count % 3 == 1 && param_.is_train ) {
	  // Read label into vector
	  std::istringstream label_stream(s);
	  while ( label_stream ) {
	    std::string label_inst;
	    if (!getline( label_stream, label_inst, ',' )) break;
	    if ( param_.is_train ) {
	      tmp.label.push_back(atof(label_inst.c_str()));
	    }
	    else {
	      tmp.data_paths.push_back(std::string(label_inst));
	    }
	  }
	  num_channels = tmp.data_paths.size();
	}
	else if ( line_count % 3 == path_idx ) {
	  // Read data_paths into vector
	  std::istringstream data_stream(s);
	  while ( data_stream ) {
	    std::string data_path;
	    if (!getline( data_stream, data_path, ',' )) break;
	    tmp.data_paths.push_back(std::string(data_path));
	  }
	  num_channels = tmp.data_paths.size();
	}
	line_count++;
      }
      data_.push_back(tmp);
    }
    dataStream.close();
  }
  
  inline void ReadInstance_(index_t idx) {
    
    std::vector<cv::Mat> images;
    // read images 
    if (param_.rand_time_slices_num == 0 && param_.num_rand_saxes == 0) {
      for ( std::string & i : data_[idx].data_paths ) {
	cv::Mat img = cv::imread(i, CV_LOAD_IMAGE_COLOR);
	CHECK(img.data) 
	  << "Could not open image: " << i << std::endl;
	images.push_back(img);
      }
	    
      // augment images with opencv 
      mshadow::Tensor<cpu, 3> spatial_tensor = (param_.data_shape.ndim() == 4) ? src_tensor_4d[0] : src_tensor_3d;    
      augmenter_->ProcessMultichannel(spatial_tensor, images, prnd_.get());
      out_.index = idx;  
    }
    else if (param_.rand_time_slices_num != 0) {
      std::uniform_int_distribution<size_t> rand_uniform_int(0, num_channels - param_.rand_time_slices_num - 1);
      index_t start = rand_uniform_int(*prnd_.get());
      for ( index_t i = 0; i < param_.rand_time_slices_num; ++i ) {
	//index_t time_idx = rand_uniform_int(*prnd_.get());
	//cv::Mat img = cv::imread(data_[idx].data_paths[time_idx], CV_LOAD_IMAGE_COLOR);
	//CHECK(img.data) 
	//  << "Could not open image: " << data_[idx].data_paths[time_idx] << std::endl;
	cv::Mat img = cv::imread(data_[idx].data_paths[start + i], CV_LOAD_IMAGE_COLOR);
	CHECK(img.data) 
	  << "Could not open image: " << data_[idx].data_paths[start + i] << std::endl;
	images.push_back(img);
      }
      mshadow::Tensor<cpu, 3> spatial_tensor = (param_.data_shape.ndim() == 4) ? src_tensor_4d[0] : src_tensor_3d;    
      augmenter_->ProcessMultichannel(spatial_tensor, images, prnd_.get());
      out_.index = idx;  
    }
    else if (param_.num_rand_saxes != 0) {
      std::uniform_int_distribution<size_t> rand_uniform_int(0, data_[idx].data_paths.size()/30 - 1);
      //mshadow::Tensor<cpu, 3> spatial_tensor = (param_.data_shape.ndim() == 4) ? src_tensor_4d[0] : src_tensor_3d;  
      for ( index_t i = 0; i < param_.num_rand_saxes; ++i ) {
	index_t start = rand_uniform_int(*prnd_.get());
	//images.clear();
	for  ( index_t j = 0; j < 30; ++j ) {
	  //std::cout << data_[idx].data_paths.size()/30 << " " << start << " " << j << std::endl;
	  cv::Mat img = cv::imread(data_[idx].data_paths[start*30 + j], CV_LOAD_IMAGE_COLOR);
	  CHECK(img.data) 
	    << "Could not open image: " << data_[idx].data_paths[start*30 + j] << std::endl;
	  images.push_back(img);
	}
	// augment images with opencv 
	//std::cout << images.size() << " " << i*30 << std::endl;
	//augmenter_->ProcessMultichannel(spatial_tensor, images, prnd_.get(), i * 30);
	//out_.index = idx;    
      }
      //std::cout << images.size() << std::endl;
      //std::cout << images.size() << " " << param_.data_shape << std::endl;
    }
    
    // augment images with opencv 
    mshadow::Tensor<cpu, 3> spatial_tensor = (param_.data_shape.ndim() == 4) ? src_tensor_4d[0] : src_tensor_3d;    
    augmenter_->ProcessMultichannel(spatial_tensor, images, prnd_.get());
    out_.index = idx;    
    
    /*
    //Generate per block mean
    block_mean_ = 0.0f;
    for (index_t i = 0; i < spatial_tensor.size(0); i++){
	block_mean_ += spatial_tensor[i];
    }
    block_mean_ /= spatial_tensor.size(0);
    for (index_t i = 0; i < spatial_tensor.size(0); i++){
	spatial_tensor[i] -= block_mean_;
    }
    */
        
    out_.data[0] = (param_.data_shape.ndim() == 4) ? TBlob(src_tensor_4d) : TBlob(src_tensor_3d);
    out_.data[1] = param_.is_train ? TBlob((real_t*)&data_[idx].label[0], param_.label_shape, cpu::kDevMask) : dummy_label; 
    
  }


  IMGListIterParam param_;
  // output instance
  DataInst out_;
  // List entrys
  std::vector<ListEntry> data_;
  // image image_augmenter_multichanner
  std::unique_ptr<ImageAugmenter_Multichannel> augmenter_;
  // internal instance counter
  index_t inst_counter_{0};
  // Random number generator
  std::unique_ptr<common::RANDOM_ENGINE> prnd_;
  // random number seed
  static const int kRandMagic = 128;
  // dummy label
  mshadow::TensorContainer<cpu, 1, real_t> dummy_label;
  // tensor for multichannel images and 2D convolution, Shape(channel, y, x)
  mshadow::Tensor<cpu, 3, real_t> src_tensor_3d;
  // tensor for 3D images and 3D convolution, Shape(channel, d, y, x)
  mshadow::Tensor<cpu, 4, real_t> src_tensor_4d;
  mshadow::Tensor<cpu, 2, real_t> block_mean_;

  // number of input channels
  index_t num_channels;
};


DMLC_REGISTER_PARAMETER(IMGListIterParam);

MXNET_REGISTER_IO_ITER(IMGListIter)
.describe("Create iterator for dataset in csv.")
.add_arguments(IMGListIterParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new IMGListIter()));
  });

}  // namespace io
}  // namespace mxnet
