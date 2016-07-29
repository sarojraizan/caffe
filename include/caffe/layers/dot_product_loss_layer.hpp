#ifndef CAFFE_DOTPRODUCT_LOSS_LAYER_HPP_
#define CAFFE_DOTPRODUCT_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Dot Product loss @f$
 *          E = 1 - \frac{1}{2N} \hat{y}_n * y_n
 *        @f$ for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-1, +1]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-1, +1]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Dot Product loss: @f$ E =
 *          1 - \frac{1}{2n} \hat{y}_n * y_n
 *        @f$
 *
 */
template <typename Dtype>
class DotProductLossLayer : public LossLayer<Dtype> {
 public:
  explicit DotProductLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DotProductLoss"; }
  
  /**
   * NOTE: In this loss layer, the groundtruth data is always expected to be the 
   * second bottom blob (thus force_backward is not set for the second bottom blob)!
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index == 0;
  }

 protected:
  /// @copydoc DotProductLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Dot Product error gradient w.r.t. the inputs.
   *
   * DotProductLossLayer \b can only compute gradients with respect to inputs bottom[0] 
   * (and will do so if propagate_down[0] is set, due to being produced by learnable parameters
   * or if force_backward is set). This layer is NOT designed to be "commutative" -- the
   * expected result is NOT the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\dpl_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \dpl_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \dpl_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            - \frac{1}{n} \sum\limits_{n=1}^N y_n
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int height_;
  int width_;
  Blob<Dtype> temp_;
  Blob<Dtype> ones_;
};

}  // namespace caffe

#endif  // CAFFE_DOTPRODUCT_LOSS_LAYER_HPP_
