#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void NormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void NormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(num_, channels_, height_, width_);
  scale_.Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void NormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CrossChannelForward_cpu(bottom, top);
}

template <typename Dtype>
void NormLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();

  // start with the constant value (load with epsilon to prevent possible zero division)
  for (int i = 0; i < scale_.count(); ++i) {
    scale_data[i] = Dtype(1e-4);
  }

  Blob<Dtype> square(1, channels_, height_, width_);
  Dtype* square_data = square.mutable_cpu_data();

  // go through the images
  for (int n = 0; n < num_; ++n) {
    // compute the square
    caffe_sqr(channels_ * height_ * width_,
        bottom_data + bottom[0]->offset(n), 
        square_data);

    // Create the first channel scale
    for (int c = 0; c < channels_; ++c) {
      caffe_add<Dtype>(height_ * width_, 
          square_data + square.offset(0, c),
          scale_data + scale_.offset(n), 
          scale_data + scale_.offset(n));
    }

    for (int c = 1; c < channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(height_ * width_,
          scale_data + scale_.offset(n, c - 1),
          scale_data + scale_.offset(n, c));
    }
  }

  // In the end, compute output
  caffe_powx<Dtype>(scale_.count(), scale_data, -0.5, top_data);
  caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
}

template <typename Dtype>
void NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  CrossChannelBackward_cpu(top, propagate_down, bottom);
}

template <typename Dtype>
void NormLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* scale_data = scale_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  Blob<Dtype> temp(num_, channels_, height_, width_);
  Dtype* temp_data = temp.mutable_cpu_data();
  caffe_set(temp.count(), Dtype(0), temp_data);

  Blob<Dtype> temp_dot(1, 1, height_, width_);
  Dtype* temp_dot_data = temp_dot.mutable_cpu_data();

  // compute bottom_diff = top_diff * scale_data^(-0.5) - dot(top_diff, top_data) / scale_data * bottom_data

  // We hack a little bit by using the diff() to store an additional result
  // compute bottom_diff = top_diff * scale_data^(-0.5)
  caffe_powx<Dtype>(scale_.count(), scale_data, -0.5, bottom_diff);
  caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);
   
  // compute temp = dot(top_diff, top_data)
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {  
      caffe_mul<Dtype>(height_ * width_,
          top_diff + top[0]->offset(n, c),
          top_data + top[0]->offset(n, c), 
          temp_dot_data);
      caffe_axpy<Dtype>(height_ * width_, 1.,
          temp_dot_data,
          temp_data + temp.offset(n));
    }
    for (int c = 1; c < channels_; ++c) {
      // copy previous channel values
      caffe_copy<Dtype>(height_ * width_,
          temp_data + temp.offset(n, c - 1),
          temp_data + temp.offset(n, c));
    }
  }
  // compute temp = temp / scale_data
  caffe_div<Dtype>(scale_.count(),
          temp_data,
          scale_data, 
          temp_data);
  // compute temp = temp * bottom_data
  caffe_mul<Dtype>(scale_.count(),
          temp_data,
          bottom_data, 
          temp_data);
  // compute bottom_diff = bottom_diff - temp
  caffe_axpy<Dtype>(scale_.count(), -1.,
          temp_data,
          bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(NormLayer);
STUB_GPU_FORWARD(NormLayer, CrossChannelForward);
STUB_GPU_BACKWARD(NormLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(NormLayer);
REGISTER_LAYER_CLASS(Norm);
}  // namespace caffe
