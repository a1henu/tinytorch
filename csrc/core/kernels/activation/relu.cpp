/**
 * @file relu.cpp
 * @brief relu operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/kernels/activation/relu.h"

template <typename Tp>
void relu_forward(Tp* output, Tp* input, size_t size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

template <typename Tp>
void relu_backward(Tp* output, Tp* input, Tp* grad, size_t size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] > 0 ? grad[i] : 0;
    }
}