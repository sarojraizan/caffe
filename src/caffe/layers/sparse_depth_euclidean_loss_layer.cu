#include <vector>
#include <sstream>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_depth_euclidean_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SetDiffZeroForMissingGTDepth(const int nthreads,
    const Dtype* const label, const int num, 
    const int height, const int width, Dtype* const diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const label_off = label + offset;
    Dtype* const diff_off = diff + offset;
    
    // set diff_ = 0 if groundtruth data is missing
    // the channel in the bottom labels blob == 0.0
    // implies that the groundtruth data is missing
    Dtype mask = label_off[0];

    if (mask == Dtype(0.0))
        diff_off[0] = Dtype(0.0);
  }
}

template <typename Dtype>
__global__ void SetBottomDiffZeroForMissingGTDepth(const int nthreads,
    const Dtype* const label, const int num, 
    const int height, const int width, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const label_off = label + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    
    // set diff_ = 0 if groundtruth data is missing
    // the channel in the bottom labels blob == 0.0
    // implies that the groundtruth data is missing
    Dtype mask = label_off[0];

    if (mask == Dtype(0.0))
        bottom_diff_off[0] = Dtype(0.0);
  }
}

template <typename Dtype>
__global__ void ComputeLogDepths(const int nthreads,
    const Dtype* const label, const int num, 
    const int height, const int width, Dtype* const logdepths) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const label_off = label + offset;
    Dtype* const logdepths_off = logdepths + offset;
    
    Dtype mask = label_off[0];

    if (mask != Dtype(0.0))
        logdepths_off[0] = log(mask)/Dtype(0.45723134);
    else
        logdepths_off[0] = Dtype(0.0);
  }
}

template <typename Dtype>
__global__ void ComputeDDiff(const int nthreads,
    const Dtype* const diff, const int num, 
    const int height, const int width, Dtype* const ddiff_x, Dtype* const ddiff_y) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const diff_off = diff + offset;
    Dtype* const ddiff_x_off = ddiff_x + offset;
    Dtype* const ddiff_y_off = ddiff_y + offset;
    
    if (w != (width - 1)) *(ddiff_x_off) = *(diff_off + 1) - *(diff_off); 
    else *(ddiff_x_off) = Dtype(0.0);
    if (h < (height - 1)) *(ddiff_y_off) = *(diff_off + width) - *(diff_off);
    else *(ddiff_y_off) = Dtype(0.0);
  }
}

template <typename Dtype>
__global__ void ComputeDiv(const int nthreads,
    const Dtype N, const int height, const int width, const Dtype* const ddiff_x, const Dtype* const ddiff_y, Dtype* const bottom_diff, Dtype top_diff_val) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * height + h) * width + w;
    const Dtype* const ddiff_x_off = ddiff_x + offset;
    const Dtype* const ddiff_y_off = ddiff_y + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    
    if (w != 0) *(bottom_diff_off) += top_diff_val*Dtype(2.0)/N*(*(ddiff_x_off-1) - *(ddiff_x_off));
    else *(bottom_diff_off) += top_diff_val*Dtype(-2.0)/N*(*ddiff_x_off);
    if (h != 0) *(bottom_diff_off) += top_diff_val*Dtype(2.0)/N*(*(ddiff_y_off - width) - *(ddiff_y_off));
    else *(bottom_diff_off) += top_diff_val*Dtype(-2.0)/N*(*ddiff_y_off);
  }
}

template <typename Dtype>
void SparseDepthEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int n_threads = num * height * width;
  ComputeLogDepths<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[1]->gpu_data(), num, height, width, logdepths_.mutable_gpu_data());

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      logdepths_.gpu_data(),
      diff_.mutable_gpu_data());

  // set diff_ = 0 if groundtruth data is missing (parallelized on gpu)
  // Note: the bottom[1] blob should contain the groundtruth labels

  // NOLINT_NEXT_LINE(whitespace/operators)
  SetDiffZeroForMissingGTDepth<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[1]->gpu_data(), num, height, width, diff_.mutable_gpu_data());

  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeDDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, diff_.gpu_data(), num, height, width, ddiff_x_.mutable_gpu_data(), ddiff_y_.mutable_gpu_data());

  Dtype dot, ddiff_x2_sum, ddiff_y2_sum;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  caffe_gpu_dot(count, diff_.gpu_data(), ones_.gpu_data(), &diff_sum);
  caffe_gpu_dot(count, ddiff_x_.gpu_data(), ddiff_x_.gpu_data(), &ddiff_x2_sum);
  caffe_gpu_dot(count, ddiff_y_.gpu_data(), ddiff_y_.gpu_data(), &ddiff_y2_sum);

  Dtype N = bottom[0]->num();
  Dtype loss = dot / N          
               + ddiff_x2_sum / N 
               + ddiff_y2_sum / N
               - diff_sum*diff_sum / (Dtype(2.0)*N*N)
               ;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SparseDepthEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   
 if (propagate_down[0]) {

     int count = bottom[0]->count();
     int num = bottom[0]->num();
     int height = bottom[0]->height();
     int width = bottom[0]->width();

     int n_threads = num * height * width;
     Dtype N = bottom[0]->num();
     Dtype top_diff_val = top[0]->cpu_diff()[0];

     caffe_gpu_axpby(
         count,              
         top_diff_val*Dtype(2.0)/N,       // a
         diff_.gpu_data(),                // x
         Dtype(0.0),                      // b
         bottom[0]->mutable_gpu_diff());  // y

     caffe_gpu_axpby(
         count,              
         top_diff_val*Dtype(-1.0)/(N*N)*diff_sum,      // a
         ones_.gpu_data(),                             // x
         Dtype(1.0),                                   // b
         bottom[0]->mutable_gpu_diff());               // y
     
     ComputeDiv<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
         n_threads, N, height, width, ddiff_x_.gpu_data(), ddiff_y_.gpu_data(), bottom[0]->mutable_gpu_diff(), top_diff_val);
     SetBottomDiffZeroForMissingGTDepth<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
         n_threads, bottom[1]->gpu_data(), num, height, width, bottom[0]->mutable_gpu_diff());	
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseDepthEuclideanLossLayer);

}  // namespace caffe
