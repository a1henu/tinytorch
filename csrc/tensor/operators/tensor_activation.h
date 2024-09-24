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

template <typename Tp>
tensor::Tensor<Tp> t_relu_f(const tensor::Tensor<Tp>& input);

template <typename Tp>
tensor::Tensor<Tp> t_relu_b(const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad);

template <typename Tp>
tensor::Tensor<Tp> t_sigmoid_f(const tensor::Tensor<Tp>& input);

template <typename Tp>
tensor::Tensor<Tp> t_sigmoid_b(const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad);

#endif