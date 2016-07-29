#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pairwise_avg_layer.hpp"

namespace caffe {

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width_)";
  num = bottom[0]->num();
  channels = bottom[0]->channels();
  height = bottom[0]->height();
  width = bottom[0]->width();
  const int dim = width*(4*height-3) - 3*height + 2;
  top[0]->Reshape(num, 1, 1, dim);
}

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CrossChannelForward_cpu(bottom, top);
}

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int width_minus_1 = width - 1;
  const int height_minus_1 = height - 1;
  Dtype sum;
  int bottom_offset, top_offset;

  // TODO: Make CPU pairwise extraction code more efficient
  // go through the images
  for (int n = 0; n < num; ++n) 
  {
    if (n == 0) top_offset = 0;
    else top_offset = n*(width*(4*height-3) - 3*height + 2);

    for (int h = 0; h < height; ++h)
    {
      for (int w = 0; w < width; ++w)
      {
	    if (n == 0) bottom_offset = h * width + w;
	    else bottom_offset = (n*channels*height + h)*width + w;
  
	    const Dtype* bottom_off = bottom_data + bottom_offset; 
	    Dtype* top_off = top_data + top_offset;

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
  }
}

template <typename Dtype>
void PairwiseAvgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(PairwiseAvgLayer);
STUB_GPU_FORWARD(PairwiseAvgLayer, CrossChannelForward);
STUB_GPU_BACKWARD(PairwiseAvgLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(PairwiseAvgLayer);
REGISTER_LAYER_CLASS(PairwiseAvg);

}  // namespace caffe
