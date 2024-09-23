/**
 * @file sigmoid.cpp
 * @brief sigmoid operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/activation/sigmoid.h"

#include "error/error.h"

template <typename Tp>
void sigmoid_forward(device::CPU* device, Tp* output, Tp* input, size_t size) {
    for (int i = 0; i < size; ++i) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
}

template <typename Tp>
void sigmoid_backward(device::CPU* device, Tp* output, Tp* input, Tp* grad, size_t size) {
    for (int i = 0; i < size; ++i) {
        Tp sigmoid = 1 / (1 + exp(-input[i]));
        output[i] = sigmoid * (1 - sigmoid) * grad[i];
    }
}

#ifndef __CUDA

template <typename Tp>
void sigmoid_forward(device::GPU* device, Tp* output, Tp* input, size_t size) {
    throw error::DeviceError("sigmoid_forward<GPU> can not be called without CUDA support.");
}

template <typename Tp>
void sigmoid_backward(device::GPU* device, Tp* output, Tp* input, Tp* grad, size_t size) {
    throw error::DeviceError("sigmoid_backward<GPU> can not be called without CUDA support.");
}

template void sigmoid_forward<float>(device::GPU* device, float* output, float* input, size_t size);
template void sigmoid_forward<double>(device::GPU* device, double* output, double* input, size_t size);

template void sigmoid_backward<float>(device::GPU* device, float* output, float* input, float* grad, size_t size);
template void sigmoid_backward<double>(device::GPU* device, double* output, double* input, double* grad, size_t size);

#endif

template void sigmoid_forward<float>(device::CPU* device, float* output, float* input, size_t size);
template void sigmoid_forward<double>(device::CPU* device, double* output, double* input, size_t size);

template void sigmoid_backward<float>(device::CPU* device, float* output, float* input, float* grad, size_t size);
template void sigmoid_backward<double>(device::CPU* device, double* output, double* input, double* grad, size_t size);