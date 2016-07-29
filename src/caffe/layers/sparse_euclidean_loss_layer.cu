#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_euclidean_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SetDiffZeroForMissingGT(const int nthreads,
    const Dtype* const label, const int num, const int channels, 
    const int height, const int width, Dtype* const diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const label_off = label + offset;
    Dtype* const diff_off = diff + offset;
    int head = 0;
    Dtype mask(0);
    
    // set diff_ = 0 if groundtruth data is missing
    // the sum of all the (three) channels in the bottom labels blob == 0
    // implies that the groundtruth data is missing
    while (head < channels) {
      mask = label_off[head * step] + mask;
      ++head;
    }
    if (mask == Dtype(0)){
        head = 0;
	while (head < channels) {
           diff_off[head * step] = Dtype(0);
	   ++head;
	}
    }
  }
}

template <typename Dtype>
void SparseEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  // set diff_ = 0 if groundtruth data is missing (parallelized on gpu)
  // Note: the bottom[1] blob should contain the groundtruth labels
  int n_threads = num * height * width;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SetDiffZeroForMissingGT<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[1]->gpu_data(), num, channels, height, width, diff_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SparseEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

 if (propagate_down[0]) {
     const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
     caffe_gpu_axpby(
         bottom[0]->count(),              // count
         alpha,                           // a
         diff_.gpu_data(),                // x
         Dtype(0),                        // b
         bottom[0]->mutable_gpu_diff());  // y
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseEuclideanLossLayer);

}  // namespace caffe
