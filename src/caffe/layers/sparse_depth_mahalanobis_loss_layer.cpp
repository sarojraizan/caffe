#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/sparse_depth_mahalanobis_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#define EPS 0.0
#define MASK_VAL -1e5

namespace caffe {

template <typename Dtype>
void SparseDepthMahalanobisLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void SparseDepthMahalanobisLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  diff_.ReshapeLike(*bottom[0]);

  if (bottom.size() >= 3) {
    Udiff_.ReshapeLike(diff_);
    UtUdiff_.ReshapeLike(diff_);
  }
}

template <typename Dtype>
void SparseDepthMahalanobisLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_count = height*width;
  Dtype* diff = diff_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  // compute the difference
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i

  // set diff_ = 0 if groundtruth data is missing
  // the channel in the bottom labels blob == -10.0
  // implies that the groundtruth data is missing
  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask = *(label + bottom[1]->offset(n) + i);
       if (mask == Dtype(MASK_VAL))
       { 
	  *(diff + bottom[1]->offset(n) + i) = Dtype(0);
       }
    }
  }

  if (bottom.size() >= 3) {  // weighted distance
    // TODO(NCB) is there a more efficient way to do this?
    Dtype reg(0);
    Dtype U;

    for (int n = 0; n < num; ++n) 
    {
       for (int h = 0; h < height; ++h) 
       {
          for (int w = 0; w < width; ++w) 
          {
              int offset = n*spatial_count + h*width + w;
 	      Dtype mask = *(label + bottom[1]->offset(n,0,h,w));
	      if (mask == Dtype(MASK_VAL))
	      { 
		  U = Dtype(1.0);
	      }
              else U = fabs(bottom[2]->cpu_data()[offset]);	      
	      
	      // Udiff
              Udiff_.mutable_cpu_data()[offset] = U*diff_.cpu_data()[offset];

	      // UtUdiff
	      UtUdiff_.mutable_cpu_data()[offset] = U*Udiff_.cpu_data()[offset];

	      // compute regularizer
	      reg += log(U + Dtype(EPS));
	  } //for w
       } //for h
    } //for n

    // difftUtUdiff
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), UtUdiff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss - reg / bottom[0]->num();
  } else {  // unweighted distance
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void SparseDepthMahalanobisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      if (bottom.size() >= 3) {
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            UtUdiff_.cpu_data(),             // a
            Dtype(0),                        // beta
            bottom[i]->mutable_cpu_diff());  // b
      } else {
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            diff_.cpu_data(),                // a
            Dtype(0),                        // beta
            bottom[i]->mutable_cpu_diff());  // b
      }
    }
  }
  if (bottom.size() >= 3 && propagate_down[2]) 
  {
    const Dtype alpha = top[0]->cpu_diff()[0]/bottom[0]->num();
    int num = bottom[0]->num();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int spatial_count = height*width;
    const Dtype* label = bottom[1]->cpu_data();

    for (int n = 0; n < num; ++n)
    {
       for (int h = 0; h < height; ++h)
       {
          for (int w = 0; w < width; ++w)
          {
              int offset = n*spatial_count + h*width + w;
	      Dtype d_loss = alpha*Udiff_.cpu_data()[offset]*diff_.cpu_data()[offset]; // the contribution from the loss
 	      Dtype mask = *(label + bottom[1]->offset(n,0,h,w));
              
              // the diagonal elements contribute to the regularizer and have an abs
              // non linearity
              Dtype d_reg = top[0]->cpu_diff()[0] / bottom[0]->num();

              d_reg *= Dtype(-1) / (fabs(bottom[2]->cpu_data()[offset]) + Dtype(EPS));
              if (bottom[2]->cpu_data()[offset] < 0) 
              {
                 d_loss *= -1;
                 d_reg *= -1;
              }

	      if (mask == Dtype(MASK_VAL))
	      { 
		  bottom[2]->mutable_cpu_diff()[offset] = Dtype(0);
	      }
              else bottom[2]->mutable_cpu_diff()[offset] = d_loss + d_reg;
               
           } // for w 
        } // for h
    } // for n
  } // if (bottom.size() >=3 ...
 }

#ifdef CPU_ONLY
STUB_GPU(SparseDepthMahalanobisLossLayer);
#endif

INSTANTIATE_CLASS(SparseDepthMahalanobisLossLayer);
REGISTER_LAYER_CLASS(SparseDepthMahalanobisLoss);
}  // namespace caffe
