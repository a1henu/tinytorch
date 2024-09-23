/**
 * @file sigmoid.h
 * @brief sigmoid operator declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_ACTIVATION_SIGMOID_H
#define CSRC_CORE_KERNELS_ACTIVATION_SIGMOID_H

# include <cstddef>

template <typename Tp>
void sigmoid_forward(Tp* output, Tp* input, size_t size);

template <typename Tp>
void sigmoid_backward(Tp* output, Tp* input, Tp* grad, size_t size);

#endif