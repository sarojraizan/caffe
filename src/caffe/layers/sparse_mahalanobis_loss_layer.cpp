#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/sparse_mahalanobis_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

const double EPS = 1e-6;

template <typename Dtype>
void SparseMahalanobisLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void SparseMahalanobisLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  diff_.ReshapeLike(*bottom[0]);

  if (bottom.size() >= 3) {
    CHECK_EQ(top.size(), 2);
    U_.Reshape(1, bottom[0]->channels(), bottom[0]->channels(), 1);
    // will only modify the upper triangle, set rest to zero
    memset(U_.mutable_cpu_data(), 0.0, U_.count() * sizeof(Dtype));
    Udiff_.ReshapeLike(diff_);
    UtUdiff_.ReshapeLike(diff_);
    // setup second top blob, regularization
    top[1]->ReshapeLike(*top[0]);
  }
}

template <typename Dtype>
void SparseMahalanobisLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
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
  // the sum of all the (three) channels in the bottom labels blob == 0
  // implies that the groundtruth data is missing
  for (int n = 0; n < num; ++n) 
  {
    for (int i = 0; i < spatial_count; ++i) 
    {
       Dtype mask(0);
       for (int c = 0; c < channels; ++c) {
	   mask = *(label + bottom[1]->offset(n,c) + i) + mask;	
       }
       if (mask == Dtype(0))
       { 
           for (int c = 0; c < channels; ++c) 
	      *(diff + bottom[1]->offset(n,c) + i) = Dtype(0);	
       }
    }
  }

  if (bottom.size() >= 3) {  // weighted distance
    // TODO(NCB) is there a more efficient way to do this?
    Dtype reg(0);

    for (int n = 0; n < num; ++n) 
    {
       for (int h = 0; h < height; ++h) 
       {
          for (int w = 0; w < width; ++w) 
          {
              Dtype mask(0);
              for (int c = 0; c < channels; ++c) 
              {
	          mask = *(label + bottom[1]->offset(n,c,h,w)) + mask;	
              }

	      // pack the upper-triangular weight matrix (Cholesky factor of information
	      // matrix)
	      int ii = 0;
              int offset1 = n*bottom[2]->channels()*spatial_count + h*width + w;
	      for (size_t i = 0; i < U_.channels(); ++i) 
              {
		for (size_t j = i; j < U_.height(); ++j) 
                {
		  Dtype val = bottom[2]->cpu_data()[offset1 + ii*spatial_count];
		  if (i == j)
                  {
                     if (mask == Dtype(0)) val = Dtype(1.0);
                     val = fabs(val);
                  }
                  U_.mutable_cpu_data()[i*U_.height() + j] = val;
		  ++ii;
		}
	      }
              
              int offset2 = n*channels*spatial_count + h*width + w;
  	      Blob<Dtype> diff_tmp(1, 1, 1, channels);
	      Blob<Dtype> Udiff_tmp(1, 1, 1, channels);
              Blob<Dtype> UtUdiff_tmp(1, 1, 1, channels);

              for (int jj = 0; jj < channels; ++jj)
                  diff_tmp.mutable_cpu_data()[jj] = diff_.cpu_data()[offset2 + jj*spatial_count];
	      
	      // Udiff
	      caffe_cpu_gemv(CblasNoTrans, U_.channels(), U_.height(), Dtype(1.0),
		  U_.cpu_data(),
		  diff_tmp.cpu_data(), Dtype(0.0),
		  Udiff_tmp.mutable_cpu_data());

              for (int jj = 0; jj < channels; ++jj)
                  Udiff_.mutable_cpu_data()[offset2 + jj*spatial_count] = Udiff_tmp.cpu_data()[jj];

	      // UtUdiff
	      caffe_cpu_gemv(CblasTrans, U_.channels(), U_.height(), Dtype(1.0),
		  U_.cpu_data(),
		  Udiff_tmp.cpu_data(),
		  Dtype(0.0), UtUdiff_tmp.mutable_cpu_data());

              for (int jj = 0; jj < channels; ++jj)
                  UtUdiff_.mutable_cpu_data()[offset2 + jj*spatial_count] = UtUdiff_tmp.cpu_data()[jj];

	      // compute regularizer
	      for (size_t i = 0; i < U_.channels(); ++i) 
		  reg += log(U_.data_at(0,i,i,0) + EPS);
	  } //for w
       } //for h
    } //for n

    // difftUtUdiff
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), UtUdiff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
    top[1]->mutable_cpu_data()[0] = Dtype(-1.0) * reg / bottom[0]->num();
  } else {  // unweighted distance
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void SparseMahalanobisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[2]->num();
    int dim = bottom[0]->channels();
    int num = bottom[0]->num();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int spatial_count = height*width;
    Dtype tmp[dim*dim];
    const Dtype* label = bottom[1]->cpu_data();

    for (int n = 0; n < num; ++n)
    {
       for (int h = 0; h < height; ++h)
       {
          for (int w = 0; w < width; ++w)
          {
              Dtype mask(0);
              for (int c = 0; c < dim; ++c) 
              {
	          mask = *(label + bottom[1]->offset(n,c,h,w)) + mask;	
              }

              int offset1 = n*dim*spatial_count + h*width + w;
	      int offset2 = n*bottom[2]->channels()*spatial_count + h*width + w;
  	      Blob<Dtype> diff_tmp(1, 1, 1, dim);
	      Blob<Dtype> Udiff_tmp(1, 1, 1, dim);

              for (int jj = 0; jj < dim; ++jj)
              {
                  diff_tmp.mutable_cpu_data()[jj] = diff_.cpu_data()[offset1 + jj*spatial_count];
		  Udiff_tmp.mutable_cpu_data()[jj] = Udiff_.cpu_data()[offset1 + jj*spatial_count];
	      }
	      caffe_cpu_gemm(CblasNoTrans,   // trans A
                     CblasNoTrans,           // trans B,
                     dim, dim, 1,            // Dimensions of A and B
                     alpha,
                     Udiff_tmp.cpu_data(),
                     diff_tmp.cpu_data(),
                     Dtype(0),
                     tmp);
	      int ii = 0;
	      for (size_t i = 0; i < U_.channels(); ++i) 
              {
                  for (size_t j = i; j < U_.height(); ++j) 
                  {
                      Dtype d_loss = tmp[i*dim+j]; // the contribution from the loss

                      // the diagonal elements contribute to the regularizer and have an abs non linearity
                      Dtype d_reg(0);
                      if (i == j) 
                      {
                         d_reg = top[1]->cpu_diff()[0] / bottom[0]->num();
                         d_reg *= Dtype(-1) / (fabs(bottom[2]->data_at(n,ii,h,w)) + EPS);
                         if (bottom[2]->data_at(n,ii,h,w) < 0) 
                         {
                            d_loss *= -1;
                            d_reg *= -1;
                         }
                      }
                      if (mask == Dtype(0)) bottom[2]->mutable_cpu_diff()[offset2 + ii*spatial_count] = Dtype(0);
                      else bottom[2]->mutable_cpu_diff()[offset2 + ii*spatial_count] = d_loss + d_reg;
                      ++ii;
                 } // for U_height 
              } // for U_channels
           } // for w 
        } // for h
    } // for n
  } // if (bottom.size() >=3 ...
 }

#ifdef CPU_ONLY
STUB_GPU(SparseMahalanobisLossLayer);
#endif

INSTANTIATE_CLASS(SparseMahalanobisLossLayer);
REGISTER_LAYER_CLASS(SparseMahalanobisLoss);
}  // namespace caffe
