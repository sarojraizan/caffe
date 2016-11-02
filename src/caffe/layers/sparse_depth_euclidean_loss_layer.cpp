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
  logdepths_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
  ddiff_x_.ReshapeLike(*bottom[0]);
  ddiff_y_.ReshapeLike(*bottom[0]);
  ones_.ReshapeLike(*bottom[0]);
  caffe_set(ones_.count(), Dtype(1.0), ones_.mutable_cpu_data());
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
  Dtype* ddiff_x = ddiff_x_.mutable_cpu_data();  
  Dtype* ddiff_y = ddiff_y_.mutable_cpu_data();  
  Dtype* logdepths = logdepths_.mutable_cpu_data(); 

  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask = *(label + bottom[1]->offset(n) + i);	
       if (mask != Dtype(0.0))
	  *(logdepths + bottom[1]->offset(n) + i) = log(mask)/Dtype(0.45723134);
       else
          *(logdepths + bottom[1]->offset(n) + i) = Dtype(0.0);
    }
  }

  caffe_sub(
      count,
      bottom_data,
      logdepths,
      diff_.mutable_cpu_data());

  // set diff_ = 0 if groundtruth data is missing
  // the channel in the bottom labels blob == 0.0
  // implies that the groundtruth data is missing
  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask = *(label + bottom[1]->offset(n) + i);	
       if (mask == Dtype(0.0))
	  *(diff + bottom[1]->offset(n) + i) = Dtype(0.0);
    }
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  diff_sum = caffe_cpu_dot(count, diff_.cpu_data(), ones_.cpu_data());
  Dtype ddiff_x2_sum = Dtype(0.0);
  Dtype ddiff_y2_sum = Dtype(0.0);

  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
        if (i%width != (width - 1))
        {
	   Dtype diff_tmp = *(diff + bottom[1]->offset(n) + i + 1) - *(diff + bottom[1]->offset(n) + i); 
           *(ddiff_x + bottom[1]->offset(n) + i) = diff_tmp;
           ddiff_x2_sum += diff_tmp*diff_tmp;
        }
	else *(ddiff_x + bottom[1]->offset(n) + i) = Dtype(0.0);
        if (i < (height-1)*width) 
        {
           Dtype diff_tmp = *(diff + bottom[1]->offset(n) + i + width) - *(diff + bottom[1]->offset(n) + i);
           *(ddiff_y + bottom[1]->offset(n) + i) = diff_tmp;
           ddiff_y2_sum += diff_tmp*diff_tmp;
        }
        else *(ddiff_y + bottom[1]->offset(n) + i) = Dtype(0.0);
    }
  }
  Dtype N = bottom[0]->count();
  Dtype loss = dot / N 
               + ddiff_x2_sum / N
               + ddiff_y2_sum / N
               - diff_sum*diff_sum / (Dtype(2.0)*N*N)
               ;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SparseDepthEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

   if (propagate_down[0]) {
     int num = bottom[0]->num();
     int height = bottom[0]->height();
     int width = bottom[0]->width();
     int spatial_count = height * width;
     const Dtype* label = bottom[1]->cpu_data();
     const Dtype* diff = diff_.cpu_data();
     const Dtype* ddiff_x = ddiff_x_.cpu_data();
     const Dtype* ddiff_y = ddiff_y_.cpu_data();
     Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); 
     Dtype top_diff_val = top[0]->cpu_diff()[0];
     Dtype N = bottom[0]->count();

     for (int n = 0; n < num; ++n) 
     {
        for (int i = 0; i < spatial_count; ++i) 
        {
           Dtype mask = *(label + bottom[1]->offset(n) + i);	
           if (mask == Dtype(0.0))
	      *(bottom_diff + bottom[0]->offset(n) + i) = Dtype(0.0);
           else 
           {
              Dtype diff_val = *(diff + bottom[0]->offset(n) + i);
	      *(bottom_diff + bottom[0]->offset(n) + i) = top_diff_val*Dtype(2.0)/N*diff_val;
              *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val*Dtype(-1.0)/(N*N)*diff_sum;
              if (i%width != 0)
                *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val*Dtype(2.0)/N*(*(ddiff_x + bottom[0]->offset(n) + i - 1) - *(ddiff_x + bottom[0]->offset(n) + i));
              else *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val*Dtype(-2.0)/N*(*(ddiff_x + bottom[0]->offset(n) + i));
              if (i >= width)
                *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val*Dtype(2.0)/N*(*(ddiff_y + bottom[0]->offset(n) + i - width) - *(ddiff_y + bottom[0]->offset(n) + i));
              else
                 *(bottom_diff + bottom[0]->offset(n) + i) += top_diff_val*Dtype(-2.0)/N*(*(ddiff_y + bottom[0]->offset(n) + i));
           }
       }
     }
   }
}

#ifdef CPU_ONLY
STUB_GPU(SparseDepthEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(SparseDepthEuclideanLossLayer);
REGISTER_LAYER_CLASS(SparseDepthEuclideanLoss);

}  // namespace caffe
