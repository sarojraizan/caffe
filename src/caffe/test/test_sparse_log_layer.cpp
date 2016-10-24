#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_log_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SparseLogLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseLogLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 1, 5, 5)),
        blob_top_(new Blob<Dtype>(10, 1, 5, 5)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.5);
    UniformFiller<Dtype>  filler(filler_param);
    //GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    // randomly set channel to 0.0 in label data
    int num = this->blob_bottom_->num();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();
    int spatial_count = height * width;
    Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();

    for (int n = 0; n < num; ++n) 
    {
        for (int i = 0; i < spatial_count; ++i)
        {
    	   if (!(caffe_rng_rand() % 2)) 
             *(bottom_data + this->blob_bottom_->offset(n) + i) = Dtype(0.0);
        }
    }

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SparseLogLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    SparseLogLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // manually compute sparse eucledean loss
    int num = this->blob_bottom_->num();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();
    int spatial_count = height * width;
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();   
    const Dtype* top_data = this->blob_top_->cpu_data();

    for (int n = 0; n < num; ++n) 
    {
      for (int i = 0; i < spatial_count; ++i) 
      {
        Dtype mask = *(bottom_data + this->blob_bottom_->offset(n) + i);
        if (mask != Dtype(0.0))
        {
            EXPECT_FLOAT_EQ(*(top_data + this->blob_bottom_->offset(n) + i), log(mask)/0.45723134);
        }
        else
        {
            EXPECT_FLOAT_EQ(*(top_data + this->blob_bottom_->offset(n) + i), 0.0);
        }
      }
    }   
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SparseLogLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseLogLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SparseLogLayerTest, TestGradient) {
}
}  // namespace caffe
