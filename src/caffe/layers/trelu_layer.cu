#include <algorithm>
#include <vector>

#include "caffe/layers/trelu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype lb, Dtype ub) {
  CUDA_KERNEL_LOOP(index, n) {
    if (in[index] < lb)
	out[index] = lb;
    else if(in[index] > ub)
	out[index] = ub;
    else 
        out[index] = in[index];
  }
}

template <typename Dtype>
void TReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype ub = this->layer_param_.trelu_param().ub();
  Dtype lb = this->layer_param_.trelu_param().lb();

  // NOLINT_NEXT_LINE(whitespace/operators)
  TReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, lb, ub);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void TReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype lb, Dtype ub) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (in_data[index] > lb && in_data[index] < ub);
  }
}

template <typename Dtype>
void TReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype ub = this->layer_param_.trelu_param().ub();
    Dtype lb = this->layer_param_.trelu_param().lb();

    // NOLINT_NEXT_LINE(whitespace/operators)
    TReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, lb, ub);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TReLULayer);
}  // namespace caffe
