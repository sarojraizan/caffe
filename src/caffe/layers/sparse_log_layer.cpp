#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_log_layer.hpp"

namespace caffe {

template <typename Dtype>
void SparseLogLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SparseLogLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_count = height * width;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_ = top[0]->mutable_cpu_data();

  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask = *(bottom_data + bottom[0]->offset(n) + i);	
       if (mask != Dtype(0.0))
	  *(top_ + bottom[0]->offset(n) + i) = log(mask)/Dtype(0.45723134);
       else
          *(top_ + bottom[0]->offset(n) + i) = Dtype(0.0);
    }
  }
}

template <typename Dtype>
void SparseLogLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(SparseLogLayer);
#endif

INSTANTIATE_CLASS(SparseLogLayer);
REGISTER_LAYER_CLASS(SparseLog);

}  // namespace caffe
