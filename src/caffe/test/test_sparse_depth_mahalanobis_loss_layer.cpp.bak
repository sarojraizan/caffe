#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_depth_mahalanobis_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SparseDepthMahalanobisLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseDepthMahalanobisLossLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_1_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype>  filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);

    // randomly set the value to -10.0 in label data
    int num = this->blob_bottom_0_->num();
    int height = this->blob_bottom_0_->height();
    int width = this->blob_bottom_0_->width();
    int spatial_count = height*width;
    Dtype* label = this->blob_bottom_1_->mutable_cpu_data();

    for (int n = 0; n < num; ++n) 
    {
        for (int i = 0; i < spatial_count; ++i) 
        {
    	    if (!(caffe_rng_rand() % 2))
              *(label + this->blob_bottom_1_->offset(n) + i) = Dtype(-10.0);
        }
    }

    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SparseDepthMahalanobisLossLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(SparseDepthMahalanobisLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseDepthMahalanobisLossLayerTest, TestForward) 
{
  typedef typename TypeParam::Dtype Dtype;
  SparseDepthMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  int height = this->blob_bottom_0_->height();
  int width = this->blob_bottom_0_->width();

  Dtype loss(0);
  for (int n = 0; n < num; ++n) 
  {
     for (int h = 0; h < height; ++h)
     {
	for (int w = 0; w < width; ++w)
        {
	    Dtype mask = this->blob_bottom_1_->cpu_data()[n*height*width + h*width + w];	

	    if (mask != Dtype(-10.0))
	    {
		Dtype diff = this->blob_bottom_0_->cpu_data()[n*height*width + h*width + w] - 
                       this->blob_bottom_1_->cpu_data()[n*height*width + h*width + w];
		loss += diff*diff;
	    }
	}
     }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-4);
}

TYPED_TEST(SparseDepthMahalanobisLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  SparseDepthMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_,0);
}


template <typename TypeParam>
class SparseDepthMahalanobisLossLayerWeightedTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseDepthMahalanobisLossLayerWeightedTest()
      : blob_bottom_0_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_1_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_2_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_reg_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.5);
    UniformFiller<Dtype>  filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);

    // randomly set the value to -10.0 in label data
    int num = this->blob_bottom_0_->num();
    int height = this->blob_bottom_0_->height();
    int width = this->blob_bottom_0_->width();
    int spatial_count = height*width;
    Dtype* label = this->blob_bottom_1_->mutable_cpu_data();

    for (int n = 0; n < num; ++n) 
    {
        for (int i = 0; i < spatial_count; ++i) 
        {
    	    if (!(caffe_rng_rand() % 2))
            {
              *(label + this->blob_bottom_1_->offset(n) + i) = Dtype(-10.0);
            }
        }
    }

    blob_bottom_vec_.push_back(blob_bottom_1_);
//     filler_param.set_mean(0.0);
//     filler_param.set_std(5.0);
//     filler = GaussianFiller<Dtype>(filler_param);

    blob_bottom_vec_.push_back(blob_bottom_2_);

    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_.push_back(blob_top_reg_);
    layer_param_.add_loss_weight(1.0);
    layer_param_.add_loss_weight(1.0);

  }
  virtual ~SparseDepthMahalanobisLossLayerWeightedTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_loss_;
    delete blob_top_reg_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_loss_;
  Blob<Dtype>* const blob_top_reg_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(SparseDepthMahalanobisLossLayerWeightedTest, TestDtypesAndDevices);

TYPED_TEST(SparseDepthMahalanobisLossLayerWeightedTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  SparseDepthMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  int height = this->blob_bottom_0_->height();
  int width = this->blob_bottom_0_->width();

  Dtype loss(0);
  Dtype reg(0);
  Dtype wdiff;
  Dtype U;

  for (int n = 0; n < num; ++n) 
  {
     for (int h = 0; h < height; ++h)
     {
	for (int w = 0; w < width; ++w)
        {
	    Dtype mask = this->blob_bottom_1_->cpu_data()[n*height*width + h*width + w];	
	    Dtype diff(0);
	    if (mask != Dtype(-10.0))
	    {
		diff = this->blob_bottom_0_->cpu_data()[n*height*width + h*width + w] - 
		       this->blob_bottom_1_->cpu_data()[n*height*width + h*width + w];
	    }

	    // build U
	    Dtype eps(1e-6);

 	    if (mask == Dtype(-10.0)) 
               U = 1; 
	    else 
               U = fabs(this->blob_bottom_2_->data_at(n, 0, h, w));

            reg += log(U + eps);

	    // apply wdiff = U*diff
	    wdiff = U*diff;

	    // compute loss
	    loss += wdiff*wdiff;
	}
     }
  }

  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-2);
  reg *= Dtype(-1) / static_cast<Dtype>(num);
  EXPECT_NEAR(this->blob_top_reg_->cpu_data()[0], reg, 1e-2);
}

TYPED_TEST(SparseDepthMahalanobisLossLayerWeightedTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  SparseDepthMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-1, 1701, 0.0, 0.1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_, 1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
}

}  // namespace caffe
