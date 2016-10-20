#include <algorithm>
#include <vector>

#include "caffe/layers/trelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void TReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype ub = this->layer_param_.trelu_param().ub();
  Dtype lb = this->layer_param_.trelu_param().lb();

  for (int i = 0; i < count; ++i) {
    top_data[i] = std::min(std::max(bottom_data[i], lb), ub);
  }
}

template <typename Dtype>
void TReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype ub = this->layer_param_.trelu_param().ub();
    Dtype lb = this->layer_param_.trelu_param().lb();

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > lb && bottom_data[i] < ub);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(TReLULayer);
#endif

INSTANTIATE_CLASS(TReLULayer);

}  // namespace caffe
