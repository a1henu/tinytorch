/**
 * @file layers.h
 * @brief Layers declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_LAYERS_LAYERS_H
#define CSRC_LAYERS_LAYERS_H

#include "tensor/tensor.h"

namespace layers {

/**
 * @brief forward function for fully connected layer
 *        - Y = XW + b
 */
template <typename Tp>
void fc_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, in_features)
    const tensor::Tensor<Tp>& weight,   // W(in_features, out_features)
    const tensor::Tensor<Tp>& bias,     // b(out_features)
    tensor::Tensor<Tp>& output          // Y(batch_size, out_features)
);

/**
 * @brief backward function for fully connected layer
 *        - dX = dY * W^T
 *        - dW = X^T * dY
 *        - db = \sum dY
 */
template <typename Tp>
void fc_backward(
    const tensor::Tensor<Tp>& input,        // X(batch_size, in_features)
    const tensor::Tensor<Tp>& weight,       // W(in_features, out_features)
    const tensor::Tensor<Tp>& bias,         // b(1, out_features)
    const tensor::Tensor<Tp>& output,       // Y(batch_size, out_features)
    tensor::Tensor<Tp>& grad_input,         // dX(batch_size, in_features)
    tensor::Tensor<Tp>& grad_weight,        // dW(in_features, out_features)
    tensor::Tensor<Tp>& grad_bias,          // db(1, out_features)
    const tensor::Tensor<Tp>& grad_output   // dY(batch_size, out_features)
);

} // namespace layers

#endif