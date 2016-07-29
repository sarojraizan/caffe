#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/grn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class GRNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GRNLayerTest()
      : epsilon_(Dtype(1e-4)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GRNLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceGRNForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void GRNLayerTest<TypeParam>::ReferenceGRNForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();

  for (int n = 0; n < blob_bottom.num(); ++n) {
    for (int c = 0; c < blob_bottom.channels(); ++c) {
	for (int h = 0; h < blob_bottom.height(); ++h) {
	  for (int w = 0; w < blob_bottom.width(); ++w) {
	    Dtype scale = 0.;
	    for (int i = 0; i < blob_bottom.channels(); ++i) {
	      Dtype value = blob_bottom.data_at(n, i, h, w);
	      scale += value * value;
	    }
	    if (scale == 0) scale+=Dtype(1e-4);

	    *(top_data + blob_top->offset(n, c, h, w)) =
	      blob_bottom.data_at(n, c, h, w) / pow(scale, 0.5);
	  }
	}
     }
  }
}

TYPED_TEST_CASE(GRNLayerTest, TestDtypesAndDevices);

TYPED_TEST(GRNLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(GRNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceGRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(GRNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);

  //for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //   std::cout << "CPU diff " << this->blob_bottom_vec_[0]->cpu_diff()[i]
  //       << std::endl;
  //}
 
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
