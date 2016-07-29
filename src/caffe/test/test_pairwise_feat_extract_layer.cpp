#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pairwise_feat_extract_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class PairwiseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PairwiseLayerTest()
      : epsilon_(Dtype(1e-4)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(1, 64, 28, 36);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PairwiseLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferencePairwiseForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void PairwiseLayerTest<TypeParam>::ReferencePairwiseForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  const int num = blob_bottom.num();
  const int channels = blob_bottom.channels();
  const int height = blob_bottom.height();
  const int width = blob_bottom.width();
  const Dtype* bottom_data = blob_bottom.cpu_data();
  const int dim = width*(4*height-3) - 3*height + 2;
  blob_top->Reshape(num, 1, 1, dim);
  Dtype* top_data = blob_top->mutable_cpu_data();

  // Purposed unoptimised/alternate code to calculate pairwise features for verification check
  for (int n = 0; n < num; ++n) 
  {
    const int top_num_offset = n * dim;

    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
    	  for (int c = 0; c < channels; ++c)
          {
	     const int bottom_offset = n * channels * height * width + c * height * width + i * width + j;
             const Dtype* bottom_pointer = bottom_data + bottom_offset;
             
             if (i == 0)
	     {
             	if (j == 0)
	     	{
 	     		const int top_offset = top_num_offset;
             		Dtype* top_pointer = top_data + top_offset;
			if (c == 0) {*(top_pointer) = 0; *(top_pointer+1) = 0; *(top_pointer+2) = 0;}
			*(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + 1))*(*(bottom_pointer) - *(bottom_pointer + 1));
			*(top_pointer+1) += (*(bottom_pointer) - *(bottom_pointer + width))*(*(bottom_pointer) - *(bottom_pointer + width));
		        *(top_pointer+2) += (*(bottom_pointer) - *(bottom_pointer + width + 1))*(*(bottom_pointer) - *(bottom_pointer + width + 1));
             	}
		else if (j == width - 1)
		{
 	     		const int top_offset = top_num_offset + 4*width - 5;
             		Dtype* top_pointer = top_data + top_offset;
			if (c == 0) {*(top_pointer) = 0; *(top_pointer+1) = 0;}
		        *(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + width - 1))*(*(bottom_pointer) - *(bottom_pointer + width - 1));	
			*(top_pointer+1) += (*(bottom_pointer) - *(bottom_pointer + width))*(*(bottom_pointer) - *(bottom_pointer + width));	
		}	  	
		else
		{
 	     		const int top_offset = top_num_offset + 4*j - 1;
             		Dtype* top_pointer = top_data + top_offset;
			if (c == 0) {*(top_pointer) = 0; *(top_pointer+1) = 0; *(top_pointer+2) = 0; *(top_pointer+3) = 0;}
			*(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + 1))*(*(bottom_pointer) - *(bottom_pointer + 1));
		        *(top_pointer+1) += (*(bottom_pointer) - *(bottom_pointer + width - 1))*(*(bottom_pointer) - *(bottom_pointer + width - 1));
			*(top_pointer+2) += (*(bottom_pointer) - *(bottom_pointer + width))*(*(bottom_pointer) - *(bottom_pointer + width));
		        *(top_pointer+3) += (*(bottom_pointer) - *(bottom_pointer + width + 1))*(*(bottom_pointer) - *(bottom_pointer + width + 1));
		}
             }
             else if (j == 0)
	     {
             	if (i == height - 1)
	     	{
 	    		const int top_offset = top_num_offset + 4*width*(height-1) - 3*height + 3;
            		Dtype* top_pointer = top_data + top_offset;
			if (c == 0) *(top_pointer) = 0;
			*(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + 1))*(*(bottom_pointer) - *(bottom_pointer + 1));
             	}
		else
		{
 	     		const int top_offset = top_num_offset + i*(4*width-3);
             		Dtype* top_pointer = top_data + top_offset;
			if (c == 0) {*(top_pointer) = 0; *(top_pointer+1) = 0; *(top_pointer+2) = 0;}
			*(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + 1))*(*(bottom_pointer) - *(bottom_pointer + 1));
			*(top_pointer+1) += (*(bottom_pointer) - *(bottom_pointer + width))*(*(bottom_pointer) - *(bottom_pointer + width));
                	*(top_pointer+2) += (*(bottom_pointer) - *(bottom_pointer + width + 1))*(*(bottom_pointer) - *(bottom_pointer + width + 1));
		}
             }
             else if (i == height - 1)
	     {
		if (j != width - 1)
		{
 	     		const int top_offset = top_num_offset + 4*width*(height-1) - 3*height + j + 3;
             		Dtype* top_pointer = top_data + top_offset;
			if (c == 0) *(top_pointer) = 0;
			*(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + 1))*(*(bottom_pointer) - *(bottom_pointer + 1));
		}
             }
             else if (j == width - 1)
	     {
 	     	const int top_offset = top_num_offset + 4*width*(i+1) - 3*i - 5;
             	Dtype* top_pointer = top_data + top_offset;
		if (c == 0) {*(top_pointer) = 0; *(top_pointer+1) = 0;}
                *(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + width - 1))*(*(bottom_pointer) - *(bottom_pointer + width - 1));
		*(top_pointer+1) += (*(bottom_pointer) - *(bottom_pointer + width))*(*(bottom_pointer) - *(bottom_pointer + width));
             }	  
             else
             {
 	     	const int top_offset = top_num_offset + i*(4*width-3) + 4*j - 1;
             	Dtype* top_pointer = top_data + top_offset;
		if (c == 0) {*(top_pointer) = 0; *(top_pointer+1) = 0; *(top_pointer+2) = 0; *(top_pointer+3) = 0;} 
		*(top_pointer) += (*(bottom_pointer) - *(bottom_pointer + 1))*(*(bottom_pointer) - *(bottom_pointer + 1));
                *(top_pointer+1) += (*(bottom_pointer) - *(bottom_pointer + width - 1))*(*(bottom_pointer) - *(bottom_pointer + width - 1));
		*(top_pointer+2) += (*(bottom_pointer) - *(bottom_pointer + width))*(*(bottom_pointer) - *(bottom_pointer + width));
                *(top_pointer+3) += (*(bottom_pointer) - *(bottom_pointer + width + 1))*(*(bottom_pointer) - *(bottom_pointer + width + 1));
             }
          }	
       }
    }
  }
}

TYPED_TEST_CASE(PairwiseLayerTest, TestDtypesAndDevices);

TYPED_TEST(PairwiseLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PairwiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  const int height = this->blob_bottom_->height();
  const int width = this->blob_bottom_->width();
  const int dim = width*(4*height-3) - 3*height + 2;
  EXPECT_EQ(this->blob_top_->width(), dim);
}

TYPED_TEST(PairwiseLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PairwiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferencePairwiseForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

}  // namespace caffe
