#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_mahalanobis_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SparseMahalanobisLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseMahalanobisLossLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(10, 3, 5, 5)),
        blob_bottom_1_(new Blob<Dtype>(10, 3, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype>  filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);

    // randomly set all channels to zero in label data
    int num = this->blob_bottom_0_->num();
    int channels = this->blob_bottom_0_->channels();
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
           for (int c = 0; c < channels; ++c)
              *(label + this->blob_bottom_1_->offset(n,c) + i) = Dtype(0);
    	}
     }
    }

    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SparseMahalanobisLossLayerTest() {
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

TYPED_TEST_CASE(SparseMahalanobisLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseMahalanobisLossLayerTest, TestForward) 
{
  typedef typename TypeParam::Dtype Dtype;
  SparseMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  const int channels = this->blob_bottom_0_->channels();
  int height = this->blob_bottom_0_->height();
  int width = this->blob_bottom_0_->width();

  Dtype loss(0);
  for (int n = 0; n < num; ++n) 
  {
     for (int h = 0; h < height; ++h)
     {
	for (int w = 0; w < width; ++w)
        {
	       Dtype mask(0);
	       for (int c = 0; c < channels; ++c) 
	       {
		   mask = this->blob_bottom_1_->cpu_data()[n*channels*height*width + c*height*width + h*width + w] + mask;	
	       }
	       for (int c = 0; c < channels; ++c) 
	       {
		  Dtype diff;
		  if (mask != Dtype(0))
		  {
		      diff = this->blob_bottom_0_->cpu_data()[n*channels*height*width + c*height*width + h*width + w] - 
                             this->blob_bottom_1_->cpu_data()[n*channels*height*width + c*height*width + h*width + w];
		      loss += diff*diff;
		  }
	       }
	}
     }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-4);
}

TYPED_TEST(SparseMahalanobisLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  SparseMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_,0);
}


template <typename TypeParam>
class SparseMahalanobisLossLayerWeightedTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseMahalanobisLossLayerWeightedTest()
      : blob_bottom_0_(new Blob<Dtype>(10, 3, 5, 5)),
        blob_bottom_1_(new Blob<Dtype>(10, 3, 5, 5)),
        blob_bottom_2_(new Blob<Dtype>(10, 6, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_reg_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype>  filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);

    // randomly set all channels to zero in label data
    int num = this->blob_bottom_0_->num();
    int channels = this->blob_bottom_0_->channels();
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
           for (int c = 0; c < channels; ++c)
              *(label + this->blob_bottom_1_->offset(n,c) + i) = Dtype(0);
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
  virtual ~SparseMahalanobisLossLayerWeightedTest() {
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

TYPED_TEST_CASE(SparseMahalanobisLossLayerWeightedTest, TestDtypesAndDevices);

TYPED_TEST(SparseMahalanobisLossLayerWeightedTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  SparseMahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  const int dim = this->blob_bottom_0_->channels();
  int channels = dim;
  int height = this->blob_bottom_0_->height();
  int width = this->blob_bottom_0_->width();
  Dtype loss(0);
  Dtype reg(0);
  Dtype *diff = new Dtype[dim];
  Dtype *wdiff = new Dtype[dim];
  Dtype *U = new Dtype[dim*dim];
  // will only modify the upper triangle, set rest to zero
  memset(U, 0.0, dim * dim * sizeof(Dtype));

  for (int n = 0; n < num; ++n) 
  {
     for (int h = 0; h < height; ++h)
     {
	for (int w = 0; w < width; ++w)
        {
	    Dtype mask(0);
            for (int c = 0; c < channels; ++c) 
		mask = this->blob_bottom_1_->cpu_data()[n*channels*height*width + c*height*width + h*width + w] + mask;	

	    for (int c = 0; c < channels; ++c) 
	    {
		if (mask != Dtype(0))
		{
		      diff[c] = this->blob_bottom_0_->cpu_data()[n*channels*height*width + c*height*width + h*width + w] - 
			        this->blob_bottom_1_->cpu_data()[n*channels*height*width + c*height*width + h*width + w];
		}
                else diff[c] = Dtype(0);
	    }

	    // build U
	    int ii = 0;
	    Dtype eps(1e-6);
	    for (int i = 0; i < dim; ++i) 
            {
	       for (int j = i; j < dim; ++j)
               {
		  if (i == j) 
                  {
                     if (mask == Dtype(0)) U[i*dim+j] = Dtype(1);
                     else U[i*dim+j] = fabs(this->blob_bottom_2_->data_at(n, ii, h, w));
		     reg += log(U[i*dim+j] + eps);
		  }  
                  else 
                  {
	             if (mask == Dtype(0)) U[i*dim+j] = Dtype(0);
                     else U[i*dim+j] = this->blob_bottom_2_->data_at(n, ii, h, w);
                  }
		  ++ii;
	       }
	    }
	    // apply wdiff = U*diff
	    caffe_cpu_gemv(CblasNoTrans, dim, dim, Dtype(1.0), U, diff, Dtype(0.0), wdiff);

	    // compute loss
	    for (int i = 0; i < dim; ++i) 
	      loss += wdiff[i]*wdiff[i];
	}
     }
  }

  delete [] diff;
  delete [] wdiff;
  delete [] U;
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-2);
  reg *= Dtype(-1) / static_cast<Dtype>(num);
  EXPECT_NEAR(this->blob_top_reg_->cpu_data()[0], reg, 1e-2);
}

TYPED_TEST(SparseMahalanobisLossLayerWeightedTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  SparseMahalanobisLossLayer<Dtype> layer(this->layer_param_);
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
