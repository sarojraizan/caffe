#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dot_product_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void DotProductLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  temp_.Reshape(num_, 1, height_, width_);
  ones_.Reshape(num_, 1, height_, width_);
  caffe_gpu_set(ones_.count(), Dtype(1), ones_.mutable_gpu_data());
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  int spatial_count = height_ * width_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype dot;

  dot = caffe_cpu_dot(
                count,
                bottom_data,
                label);

  // increment dot product by 1 if groundtruth data is missing 
  // the sum of all the (three) channels in the bottom labels blob == 0
  // implies that the groundtruth data is missing
  for (int n = 0; n < num_; ++n) {
    for (int i = 0; i < spatial_count; ++i) {
       Dtype mask(0);
       for (int c = 0; c < channels_; ++c) {
	   mask = *(label + bottom[1]->offset(n,c) + i) + mask;	
       }
       if (mask == Dtype(0)){ 
           dot += 1;
       }
    }
  }

  Dtype loss = spatial_count - dot / num_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype sign(-1);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / num_;
    caffe_cpu_axpby(
    bottom[0]->count(),              // count
    alpha,                           // a
    bottom[1]->cpu_data(),           // x
    Dtype(0),                        // b
    bottom[0]->mutable_cpu_diff());  // y
  }
}

#ifdef CPU_ONLY
STUB_GPU(DotProductLossLayer);
#endif

INSTANTIATE_CLASS(DotProductLossLayer);
REGISTER_LAYER_CLASS(DotProductLoss);

}  // namespace caffe
