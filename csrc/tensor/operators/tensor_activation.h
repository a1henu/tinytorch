/**
 * @file tensor_activation.h
 * @brief Tensor activation function declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_TENSOR_OPERATOR_TENSOR_ACTIVATION_H
#define CSRC_TENSOR_OPERATOR_TENSOR_ACTIVATION_H

#include "tensor/tensor.h"

namespace tensor {

template <typename Tp>
void relu_forward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input);

template <typename Tp>
void relu_backward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad);

template <typename Tp>
void sigmoid_forward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input);

template <typename Tp>
void sigmoid_backward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad);

} // namespace tensor

#endif