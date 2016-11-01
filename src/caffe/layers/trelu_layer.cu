#include <algorithm>
#include <vector>

#include "caffe/layers/trelu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TReLUForward(const int n, const Dtype* bottom_data, Dtype* top_data,
    Dtype lb, Dtype ub) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom_data[index] < lb)
	top_data[index] = lb;
    else if(bottom_data[index] > ub)
	top_data[index] = ub;
    else 
        top_data[index] = bottom_data[index];
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
__global__ void TReLUBackward(const int n, const Dtype* top_diff,
    const Dtype* bottom_data, Dtype* bottom_diff, Dtype lb, Dtype ub) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom_data[index] > lb && bottom_data[index] < ub) bottom_diff[index] = top_diff[index];
    else bottom_diff[index] = Dtype(0.0);
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
