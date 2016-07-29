#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/norm_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void NormFillScale(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    Dtype accum_scale = 0;

    // square and sum the data in each channel
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }

    head = 0;

    // add epsilon to prevent division by zero
    if (accum_scale == 0) accum_scale+=Dtype(1e-4);

    // save accumulated data in scale blob
    while (head < channels) {
      scale_off[head * step] = accum_scale;
      ++head;
    }
  }
}


template <typename Dtype>
void NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CrossChannelForward_gpu(bottom, top);
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void NormComputeOutput(const int nthreads, const Dtype* const in,
    const Dtype* const scale, Dtype* const out, Dtype neg_beta) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], neg_beta);
  }
}

template <typename Dtype>
void NormLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  Dtype neg_beta = -0.5;
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  NormFillScale<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_, channels_, height_, width_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  NormComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, top_data, neg_beta);
  CUDA_POST_KERNEL_CHECK;
}

template void NormLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void NormLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);


template <typename Dtype>
void NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  CrossChannelBackward_gpu(top, propagate_down, bottom);
}

template <typename Dtype>
__global__ void NormComputeDiff(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data, 
    const Dtype* const scale, const Dtype* const top_diff, const int num, const int channels, 
    const int height, const int width, Dtype* const bottom_diff, Dtype neg_beta) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    Dtype dot = Dtype(0);
    int head = 0;

    // compute bottom_diff = top_diff * scale_data^(-0.5) - dot(top_diff, top_data) / scale_data * bottom_data 
    // first compute dot = dot(top_diff, top_data)
    while (head < channels) {

      dot = dot + (top_diff_off[head * step] * top_off[head * step]);
      ++head;
    }
    head = 0;
    // now compute bottom_diff = top_diff * scale_data^(-0.5) - dot / scale_data * bottom_data
    while (head < channels) {

      bottom_diff_off[head * step] = (top_diff_off[head * step] * pow(scale_off[head * step], neg_beta)) -
                                     (bottom_off[head * step] * dot / scale_off[head * step]);
      ++head;
    }
  }
}

template <typename Dtype>
void NormLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int n_threads = num_ * height_ * width_;
  Dtype neg_beta = -0.5;
  // NOLINT_NEXT_LINE(whitespace/operators)
  NormComputeDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(), scale_.gpu_data(), top[0]->gpu_diff(), 
      num_, channels_, height_, width_, bottom[0]->mutable_gpu_diff(), neg_beta);
}
template void NormLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void NormLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);

INSTANTIATE_LAYER_GPU_FUNCS(NormLayer);
}  // namespace caffe
