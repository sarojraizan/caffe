#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_canonical_gaussian_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace caffe {

template <typename TypeParam>
class SparseCanonicalGaussianLossLayerWeightedTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseCanonicalGaussianLossLayerWeightedTest()
      : blob_bottom_0_(new Blob<Dtype>(5, 3, 5, 5)),
        blob_bottom_1_(new Blob<Dtype>(5, 3, 5, 5)),
        blob_bottom_2_(new Blob<Dtype>(5, 6, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
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
              *(label + this->blob_bottom_1_->offset(n,c) + i) = Dtype(0.0);
    	}
     }
    }

    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);

    blob_top_vec_.push_back(blob_top_loss_);
    layer_param_.add_loss_weight(1.0);

  }

  virtual ~SparseCanonicalGaussianLossLayerWeightedTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_loss_;
 }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(SparseCanonicalGaussianLossLayerWeightedTest, TestDtypesAndDevices);

TYPED_TEST(SparseCanonicalGaussianLossLayerWeightedTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  SparseCanonicalGaussianLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  const int channels = this->blob_bottom_0_->channels();
  int height = this->blob_bottom_0_->height();
  int width = this->blob_bottom_0_->width();
  int spatial_count = height*width;
  const Dtype* label = this->blob_bottom_1_->cpu_data(); 
  MatrixXf U_n = MatrixXf::Zero(channels,channels);
  MatrixXf UtU_n(channels, channels);
  VectorXf theta_n(channels);
  VectorXf y_n(channels);
  Dtype loss (0.0);

  // will only modify the upper triangle, set rest to zero
  for (int n = 0; n < num; ++n) 
  {
     for (int h = 0; h < height; ++h)
     {
	for (int w = 0; w < width; ++w)
        {

              Dtype mask(0);
              for (int c = 0; c < channels; ++c) 
              {
	          mask = *(label + this->blob_bottom_1_->offset(n,c,h,w)) + mask;	
              }
	      
              if (mask != 0) {
	  	          
	      	// pack the upper-triangular weight matrix (Cholesky factor of information matrix)
		int ii = 0;
	  	int offset1 = n*this->blob_bottom_2_->channels()*spatial_count + h*width + w;
		int offset2 = n*channels*spatial_count + h*width + w;
	
		for (size_t i = 0; i < channels; ++i) 
		{
		   for (size_t j = i; j < channels; ++j) 
		   {
		        U_n(i,j) = this->blob_bottom_2_->cpu_data()[offset1 + ii*spatial_count];
			++ii;
		   }
		}

		for (size_t i = 0; i < channels; ++i) 
		{
		   theta_n[i] = this->blob_bottom_0_->cpu_data()[offset2 + i*spatial_count];
		}
		
		for (size_t i = 0; i < channels; ++i) 
		{
		   y_n[i] = this->blob_bottom_1_->cpu_data()[offset2 + i*spatial_count];
		}

		UtU_n = U_n.transpose()*U_n;
	
	        // compute log determinent of UtU_n
		Dtype UtU_n_log_det(0.0);
	        for (size_t i = 0; i < channels; ++i) 
		   UtU_n_log_det += Dtype(2.0)*log(U_n(i,i));
		loss += -theta_n.transpose()*y_n + Dtype(0.5)*(y_n.transpose()*UtU_n*y_n - 
                                                   UtU_n_log_det + 
                                                   theta_n.transpose()*UtU_n.inverse()*theta_n + 
                                                   Dtype(2.0)*log(Dtype(2.0)*M_PI));	      

            }
	}
     }
  }
  Dtype N = this->blob_bottom_0_->num();
  loss = loss / N / Dtype(2.0);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-2);
}

TYPED_TEST(SparseCanonicalGaussianLossLayerWeightedTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  SparseCanonicalGaussianLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-1, 1701, 0.0, 0.1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_, 2);
}
}  // namespace caffe
