#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_depth_euclidean_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SparseDepthEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SparseDepthEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_count = height * width;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* diff = diff_.mutable_cpu_data();  

  caffe_sub(
      count,
      bottom_data,
      label,
      diff_.mutable_cpu_data());

  // set diff_ = 0 if groundtruth data is missing
  // the channel in the bottom labels blob == -10.0
  // implies that the groundtruth data is missing
  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask = *(label + bottom[1]->offset(n) + i);	
       if (mask == Dtype(-10.0))
	  *(diff + bottom[1]->offset(n) + i) = Dtype(0);
    }
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SparseDepthEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   if (propagate_down[0]) {
     const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
     caffe_cpu_axpby(
         bottom[0]->count(),              // count
         alpha,                           // a
         diff_.cpu_data(),                // x
         Dtype(0),                        // b
         bottom[0]->mutable_cpu_diff());  // y
   }
}

#ifdef CPU_ONLY
STUB_GPU(SparseDepthEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(SparseDepthEuclideanLossLayer);
REGISTER_LAYER_CLASS(SparseDepthEuclideanLoss);

}  // namespace caffe
