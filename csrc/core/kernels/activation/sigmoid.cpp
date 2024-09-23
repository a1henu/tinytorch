/**
 * @file sigmoid.cpp
 * @brief sigmoid operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/activation/sigmoid.h"

template <typename Tp>
void sigmoid_forward(Tp* output, Tp* input, size_t size) {
    for (int i = 0; i < size; ++i) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
}

template <typename Tp>
void sigmoid_backward(Tp* output, Tp* input, Tp* grad, size_t size) {
    for (int i = 0; i < size; ++i) {
        Tp sigmoid = 1 / (1 + exp(-input[i]));
        output[i] = sigmoid * (1 - sigmoid) * grad[i];
    }
}