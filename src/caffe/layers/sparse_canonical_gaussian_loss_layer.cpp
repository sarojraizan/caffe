#include <algorithm>
#include <vector>
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/layers/sparse_canonical_gaussian_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace caffe {

const double EPS = 1e-6;

template <typename Dtype>
void SparseCanonicalGaussianLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void SparseCanonicalGaussianLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void SparseCanonicalGaussianLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_count = height*width;
  const Dtype* label = bottom[1]->cpu_data(); 
  MatrixXf U_n = MatrixXf::Zero(channels,channels);
  MatrixXf UtU_n(channels, channels);
  VectorXf theta_n(channels);
  VectorXf y_n(channels);
  Dtype loss (0.0);

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
	      
              if (mask != 0) {
	  	          
	      	// pack the upper-triangular weight matrix (Cholesky factor of information matrix)
		int ii = 0;
	  	int offset1 = n*bottom[2]->channels()*spatial_count + h*width + w;
		int offset2 = n*channels*spatial_count + h*width + w;
		
		for (size_t i = 0; i < channels; ++i) 
		{
		   for (size_t j = i; j < channels; ++j) 
		   {
		        U_n(i,j) = bottom[2]->cpu_data()[offset1 + ii*spatial_count];
			++ii;
		   }
		}

		for (size_t i = 0; i < channels; ++i) 
		{
		   theta_n[i] = bottom[0]->cpu_data()[offset2 + i*spatial_count];
		}
		
		for (size_t i = 0; i < channels; ++i) 
		{
		   y_n[i] = bottom[1]->cpu_data()[offset2 + i*spatial_count];
		}

		UtU_n = U_n.transpose()*U_n;
	
	        // compute log determinent of UtU_n
		Dtype UtU_n_log_det(0.0);
	        for (size_t i = 0; i < channels; ++i) 
		   UtU_n_log_det += Dtype(2.0)*log(U_n(i,i));
		loss += -theta_n.transpose()*y_n + Dtype(0.5)*(y_n.transpose()*UtU_n*y_n - 
                                                   UtU_n_log_det + 
                                                   theta_n.transpose()*UtU_n.inverse()*theta_n + 
                                                   Dtype(2.0)*log(Dtype(2.0)*M_PI));	      
            }
	  } //for w
        } //for h
    } //for n

    loss = loss / num / Dtype(2.0);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SparseCanonicalGaussianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{

    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[2]->num();
    int channels = bottom[0]->channels();
    int num = bottom[0]->num();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int spatial_count = height*width;
    const Dtype* label = bottom[1]->cpu_data();
    MatrixXf U_n = MatrixXf::Zero(channels,channels);
    MatrixXf U_inv_n(channels, channels);
    MatrixXf UtU_n(channels, channels);
    MatrixXf UtU_inv_n(channels, channels);
    VectorXf theta_n(channels);
    VectorXf y_n(channels);
    MatrixXf d_loss_U_n(channels, channels);
    VectorXf d_loss_theta_n(channels);
 
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

              if (mask != 0) {
	  	          
	      	// pack the upper-triangular weight matrix (Cholesky factor of information matrix)
		int ii = 0;
	  	int offset1 = n*bottom[2]->channels()*spatial_count + h*width + w;
		int offset2 = n*channels*spatial_count + h*width + w;
		
		for (size_t i = 0; i < channels; ++i) 
		{
		   for (size_t j = i; j < channels; ++j) 
		   {
		        U_n(i,j) = bottom[2]->cpu_data()[offset1 + ii*spatial_count];
			++ii;
		   }
		}

		for (size_t i = 0; i < channels; ++i) 
		{
		   theta_n[i] = bottom[0]->cpu_data()[offset2 + i*spatial_count];
		}
		
		for (size_t i = 0; i < channels; ++i) 
		{
		   y_n[i] = bottom[1]->cpu_data()[offset2 + i*spatial_count];
		}
		
		U_inv_n = U_n.inverse();
		UtU_n = U_n.transpose()*U_n;
                UtU_inv_n = UtU_n.inverse();	
                
                d_loss_U_n = alpha*Dtype(0.5)*(Dtype(2.0)*U_n*y_n*y_n.transpose() - 
                                               Dtype(2.0)*U_inv_n.transpose() - 
                                               UtU_inv_n*theta_n*theta_n.transpose()*UtU_inv_n*(U_n + U_n.transpose()));  
  
		d_loss_theta_n = alpha*Dtype(0.5)*(UtU_inv_n*theta_n - y_n);
		ii = 0;              
		for (size_t i = 0; i < channels; ++i) 
		{
		   for (size_t j = i; j < channels; ++j) 
		   {
			bottom[2]->mutable_cpu_diff()[offset1 + ii*spatial_count] = d_loss_U_n(i,j);
			++ii;
		   }
		}

		for (size_t i = 0; i < channels; ++i) 
		{
		   bottom[0]->mutable_cpu_diff()[offset2 + i*spatial_count] = d_loss_theta_n[i];	
		}	      
            }
            else {

        	int ii = 0;
	  	int offset1 = n*bottom[2]->channels()*spatial_count + h*width + w;
		int offset2 = n*channels*spatial_count + h*width + w;

		for (size_t i = 0; i < channels; ++i) 
		{
		   for (size_t j = i; j < channels; ++j) 
		   {
			bottom[2]->mutable_cpu_diff()[offset1 + ii*spatial_count] = Dtype(0.0);
			++ii;
		   }
		}

		for (size_t i = 0; i < channels; ++i) 
		{
		   bottom[0]->mutable_cpu_diff()[offset2 + i*spatial_count] = Dtype(0.0);	
		}

            }
           } // for w 
        } // for h
    } // for n
}

#ifdef CPU_ONLY
STUB_GPU(SparseCanonicalGaussianLossLayer);
#endif

INSTANTIATE_CLASS(SparseCanonicalGaussianLossLayer);
REGISTER_LAYER_CLASS(SparseCanonicalGaussianLoss);
}  // namespace caffe
