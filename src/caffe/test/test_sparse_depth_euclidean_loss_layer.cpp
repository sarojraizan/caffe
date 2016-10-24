#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_depth_euclidean_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SparseDepthEuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseDepthEuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.5);
    UniformFiller<Dtype>  filler(filler_param);
    //GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);

    // randomly set channel to 0.0 in label data
    int num = this->blob_bottom_data_->num();
    int height = this->blob_bottom_data_->height();
    int width = this->blob_bottom_data_->width();
    int spatial_count = height * width;
    Dtype* label = this->blob_bottom_label_->mutable_cpu_data();

    for (int n = 0; n < num; ++n) 
    {
        for (int i = 0; i < spatial_count; ++i)
        {
    	   if (!(caffe_rng_rand() % 2)) 
             *(label + this->blob_bottom_label_->offset(n) + i) = Dtype(0.0);
        }
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SparseDepthEuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    SparseDepthEuclideanLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // manually compute sparse eucledean loss
    int num = this->blob_bottom_data_->num();
    int height = this->blob_bottom_data_->height();
    int width = this->blob_bottom_data_->width();
    int spatial_count = height * width;
    const Dtype* label = this->blob_bottom_label_->cpu_data();
    const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
    Dtype dot(0);
    
    // compute dot product if groundtruth data is present
    for (int n = 0; n < num; ++n) 
    {
      for (int i = 0; i < spatial_count; ++i) 
      {
        Dtype mask = *(label + this->blob_bottom_label_->offset(n) + i);
        if (mask != Dtype(0.0))
        {
            const Dtype label_ = *(label + this->blob_bottom_label_->offset(n) + i);
            Dtype diff = *(bottom_data + this->blob_bottom_data_->offset(n) + i) - log(label_)/0.45723134;
            dot += diff*diff;
        }
      }
    }

    // compute loss
    Dtype loss = dot / num / Dtype(2);

    // check sparse euclidean loss
    EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-4);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SparseDepthEuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseDepthEuclideanLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SparseDepthEuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SparseDepthEuclideanLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check loss derivative with respect to first blob in bottom vector 
  // (we expect the groundtruth data to be always the second blob!)
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
}  // namespace caffe
