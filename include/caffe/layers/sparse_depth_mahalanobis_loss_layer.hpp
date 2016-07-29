#ifndef CAFFE_SPARSEDEPTHMAHALANOBIS_LOSS_LAYER_HPP_
#define CAFFE_SPARSEDEPTHMAHALANOBIS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the mahalanobis loss on depth data where groundtruth is present
 */
template <typename Dtype>
class SparseDepthMahalanobisLossLayer : public LossLayer<Dtype> {
 public:
  explicit SparseDepthMahalanobisLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // SparseDepthMahalanobisLoss takes 2-3 bottom Blobs
  // The first two are required mu_estimated and mu_true
  // The third is optional Lambda_estimated (inverse covariance of the estimate)
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  // SparseDepthMahalanobisLoss outputs 1-2 blobs
  // The first is the sum of squared weighted distance
  // The second is the negative determinent of the information matrix
  //    this only is output if three bottom blobs are provided
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual inline const char* type() const { return "SparseDepthMahalanobisLoss"; }

 protected:
  /// @copydoc SparseDepthMahalanobisLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the mahalanobis error gradient w.r.t. the inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;  // cached for backward pass
  Blob<Dtype> Udiff_;  // cached for backward pass
  Blob<Dtype> UtUdiff_;  // cached for backward pass
};


}  // namespace caffe

#endif  // CAFFE_SPARSEDEPTHMAHALANOBIS_LOSS_LAYER_HPP_
