#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/trelu_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define EPS 1e-2
#define UB 10
#define LB 0.1
#define EXTREME_VAL -1e-6 

namespace caffe {

template <typename TypeParam>
class TReLULayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TReLULayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 1, 3, 3)),
        blob_top_data_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype>  filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_data_);
    
    int count = blob_bottom_data_->count();
    Dtype* bottom_data = blob_bottom_data_->mutable_cpu_data();

    for (int i = 0; i < count; ++i) {
       Dtype val = *(bottom_data + i); 
       if ((val > (Dtype(LB) - Dtype(EPS)) && val < (Dtype(LB) + Dtype(EPS))) ||
           (val > (Dtype(UB) - Dtype(EPS)) && val < (Dtype(UB) + Dtype(EPS)))) 
	   *(bottom_data + i) = EXTREME_VAL;
    }
  }
  virtual ~TReLULayerTest() {
    delete blob_bottom_data_;
    delete blob_top_data_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    TReLUParameter* trelu_param = layer_param.mutable_trelu_param();
    trelu_param->set_ub(Dtype(UB));
    trelu_param->set_lb(Dtype(LB));
    TReLULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // manually compute sparse eucledean loss
    int num = this->blob_bottom_data_->num();
    int height = this->blob_bottom_data_->height();
    int width = this->blob_bottom_data_->width();
    int spatial_count = height * width;
    const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
    const Dtype* top_data = this->blob_top_data_->cpu_data();

    for (int n = 0; n < num; ++n) {
     for (int i = 0; i < spatial_count; ++i) 
     {
       Dtype top_val = *(bottom_data + this->blob_bottom_data_->offset(n) + i);	
       if (top_val < trelu_param->lb()) 
           top_val = trelu_param->lb();
       if (top_val > trelu_param->ub()) 
           top_val = trelu_param->ub();
       EXPECT_EQ(*(top_data + + this->blob_top_data_->offset(n) + i), top_val);	
     }
    }
  }
  
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TReLULayerTest, TestDtypesAndDevices);

TYPED_TEST(TReLULayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(TReLULayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  TReLUParameter* trelu_param = layer_param.mutable_trelu_param();
  trelu_param->set_ub(Dtype(UB));
  trelu_param->set_lb(Dtype(LB));
  TReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check loss derivative with respect to first blob in bottom vector 
  // (we expect the groundtruth data to be always the second blob!)
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
}  // namespace caffe
