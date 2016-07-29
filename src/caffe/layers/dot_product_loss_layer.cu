#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dot_product_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SetTempForMissingGT(const int nthreads, const Dtype* const bottom,
    const Dtype* const label, const int num, const int channels, 
    const int height, const int width, Dtype* const temp) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int offset_temp = (n * height + h) * width + w;
    const int step = height * width;
    const Dtype* const label_off = label + offset;
    const Dtype* const bottom_off = bottom + offset;
    Dtype* const temp_off = temp + offset_temp;
    int head = 0;
    Dtype mask(0);
    
    // calculate dot product if groundtruth is present, else set dot product to 1
    // the sum of all the (three) channels in the bottom labels blob == 0
    // implies that the groundtruth data is missing
    while (head < channels) {
      mask = label_off[head * step] + mask;
      ++head;
    }
    if (mask != Dtype(0)){
        temp_off[0] = bottom_off[0] * label_off[0];
        head = 1;
	while (head < channels) {
           temp_off[0] += bottom_off[head * step] * label_off[head * step];
	   ++head;
	}
    }
    else
    {
        temp_off[0] = 1;
    }
  }
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* temp = temp_.mutable_gpu_data();
  const Dtype* ones = ones_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  // calculate dot product if groundtruth is present, 
  // else set dot product to 1 (parallelized on gpu)
  // Note: the bottom[1] blob should contain the groundtruth labels
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SetTempForMissingGT<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, label, num_, channels_, height_, width_, temp);

  Dtype dot;
  caffe_gpu_dot(ones_.count(), ones, temp, &dot);
  Dtype loss = (height_ * width_) - dot / num_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype sign(-1);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / num_;
    caffe_gpu_axpby(
        bottom[0]->count(),              // count
        alpha,                           // a
        bottom[1]->gpu_data(),           // x
        Dtype(0),                        // b
        bottom[0]->mutable_gpu_diff());  // y
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProductLossLayer);
}  // namespace caffe
