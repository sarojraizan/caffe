#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pairwise_avg_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PairwiseAvgFeatureExtract(const int nthreads, const Dtype* const bottom,
    const int num, const int channels, const int height,
    const int width, Dtype* const top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    /* uncomment following lines and comment sequent two lines to handle n > 0 */
    /* commented for speed */
    //const int n = index / width / height;
    //int bottom_offset, top_offset;
    //if (n == 0)
    //{
    // 	bottom_offset = h*width + w;
    //	top_offset = 0;  
    //}
    //else
    //{
    //	bottom_offset = (n*channels*height + h)*width + w;  
    //	top_offset = n*(width*(4*height-3) - 3*height + 2);
    //}
    const int bottom_offset = h*width + w;
    const int top_offset = 0;
    const Dtype* const bottom_off = bottom + bottom_offset; 
    Dtype* const top_off = top + top_offset; 
    const int width_minus_1 = width - 1;
    const int height_minus_1 = height - 1;
    Dtype sum;

    if (h != 0 && w != 0 && h != height_minus_1 && w != width_minus_1)
    {
             	Dtype* top_pointer = top_off + h*(4*width-3) + 4*w - 1;
                const Dtype* const bottom_pointer = bottom_off; 
                sum = *(bottom_pointer) + *(bottom_pointer + 1);		*(top_pointer)   = sum*Dtype(0.5);
                sum = *(bottom_pointer) + *(bottom_pointer + width_minus_1);	*(top_pointer+1) = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width);		*(top_pointer+2) = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width + 1);	*(top_pointer+3) = sum*Dtype(0.5);
    }
    else if (h == 0)
    {
        if (w != 0 && w != width_minus_1)
	{
             	Dtype* top_pointer = top_off + 4*w - 1;
                const Dtype* const bottom_pointer = bottom_off;
                sum = *(bottom_pointer) + *(bottom_pointer + 1);		*(top_pointer)   = sum*Dtype(0.5);
                sum = *(bottom_pointer) + *(bottom_pointer + width_minus_1);	*(top_pointer+1) = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width);		*(top_pointer+2) = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width + 1);	*(top_pointer+3) = sum*Dtype(0.5);
        }
        else if (w == 0)
        {
	     	Dtype* top_pointer = top_off;
                const Dtype* const bottom_pointer = bottom_off;
                sum = *(bottom_pointer) + *(bottom_pointer + 1);		*(top_pointer)   = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width);		*(top_pointer+1) = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width + 1);	*(top_pointer+2) = sum*Dtype(0.5);
        }	  	
        else
        {
             	Dtype* top_pointer = top_off + 4*width - 5;
                const Dtype* const bottom_pointer = bottom_off; 
                sum = *(bottom_pointer) + *(bottom_pointer + width_minus_1);	*(top_pointer)   = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width);		*(top_pointer+1) = sum*Dtype(0.5);
        }
    }
    else if (w == 0 && h != height_minus_1)
    {
             	Dtype* top_pointer = top_off + h*(4*width-3);
                const Dtype* const bottom_pointer = bottom_off; 
                sum = *(bottom_pointer) + *(bottom_pointer + 1);		*(top_pointer)   = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width);		*(top_pointer+1) = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width + 1);	*(top_pointer+2) = sum*Dtype(0.5);
    }
    else if (h != height_minus_1 && w == width_minus_1)
    {
             	Dtype* top_pointer = top_off + 4*width*(h+1) - 3*h - 5;
                const Dtype* const bottom_pointer = bottom_off;
                sum = *(bottom_pointer) + *(bottom_pointer + width_minus_1);	*(top_pointer)   = sum*Dtype(0.5);
		sum = *(bottom_pointer) + *(bottom_pointer + width);		*(top_pointer+1) = sum*Dtype(0.5);

    }
    else if (w != width_minus_1)
    {
             	Dtype* top_pointer = top_off + 4*width*(height_minus_1) - 3*height + w + 3;
                const Dtype* const bottom_pointer = bottom_off; 
                sum = *(bottom_pointer) + *(bottom_pointer + 1);		*(top_pointer)   = sum*Dtype(0.5);
    }
  }
}

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CrossChannelForward_gpu(bottom, top);
}

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num * height * width;

  // NOLINT_NEXT_LINE(whitespace/operators)
  PairwiseAvgFeatureExtract<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num, channels, height, width, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template void PairwiseAvgLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void PairwiseAvgLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseAvgLayer);

}  // namespace caffe
