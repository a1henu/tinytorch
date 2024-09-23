/**
 * @file relu.cpp
 * @brief relu operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"
#include "core/kernels/activation/relu.h"

#include "error/error.h"

template <typename Tp>
void relu_forward(device::CPU* device, Tp* output, Tp* input, size_t size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

template <typename Tp>
void relu_backward(device::CPU* device, Tp* output, Tp* input, Tp* grad, size_t size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] > 0 ? grad[i] : 0;
    }
}

#ifndef __CUDA

template <typename Tp>
void relu_forward(device::GPU* device, Tp* output, Tp* input, size_t size) {
    throw error::DeviceError("relu_forward<GPU> can not be called without CUDA support.");
}

template <typename Tp>
void relu_backward(device::GPU* device, Tp* output, Tp* input, Tp* grad, size_t size) {
    throw error::DeviceError("relu_backward<GPU> can not be called without CUDA support.");
}

template void relu_forward<float>(device::GPU* device, float* output, float* input, size_t size);
template void relu_forward<double>(device::GPU* device, double* output, double* input, size_t size);

template void relu_backward<float>(device::GPU* device, float* output, float* input, float* grad, size_t size);
template void relu_backward<double>(device::GPU* device, double* output, double* input, double* grad, size_t size);

#endif

template void relu_forward<float>(device::CPU* device, float* output, float* input, size_t size);
template void relu_forward<double>(device::CPU* device, double* output, double* input, size_t size);

template void relu_backward<float>(device::CPU* device, float* output, float* input, float* grad, size_t size);
template void relu_backward<double>(device::CPU* device, double* output, double* input, double* grad, size_t size);