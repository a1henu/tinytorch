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

/**
 * @brief forward function for softmax layer
 *        - Y = softmax(X)
 */
template <typename Tp>
void softmax_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    tensor::Tensor<Tp>& output          // Y(batch_size, num_classes)
);

/**
 * @brief loss function for cross entropy
 *       - loss = -\sum y_i * log(p_i)
 */
template <typename Tp>
void cross_entropy_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<int>& target,  // t(batch_size)
    tensor::Tensor<Tp>& output          // z(1)
);

/**
 * @brief backward function for cross entropy
 *        - dX_i = p_i - y_i
 */
template <typename Tp>
void cross_entropy_backward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<int>& target,  // t(batch_size)
    tensor::Tensor<Tp>& grad            // dX(batch_size, num_classes)
);


} // namespace layers

#endif