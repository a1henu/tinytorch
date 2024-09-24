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

namespace ops {

template <typename Tp>
struct sigmoid_forward<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        Tp* input, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = 1 / (1 + exp(-input[i]));
        }
    }
};

template <typename Tp>
struct sigmoid_backward<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        Tp* input, 
        Tp* grad, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            Tp sigmoid = 1 / (1 + exp(-input[i]));
            output[i] = sigmoid * (1 - sigmoid) * grad[i];
        }
    }
};

#ifndef __CUDA

template <typename Tp>
struct sigmoid_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input, 
        size_t size
    ) {
        throw error::DeviceError("sigmoid_forward<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct sigmoid_backward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input, 
        Tp* grad, 
        size_t size
    ) {
        throw error::DeviceError("sigmoid_backward<GPU> can not be called without CUDA support.");
    }
};

template struct sigmoid_forward<float, device::GPU>;
template struct sigmoid_forward<double, device::GPU>;

template struct sigmoid_backward<float, device::GPU>;
template struct sigmoid_backward<double, device::GPU>;

#endif

template struct sigmoid_forward<float, device::CPU>;
template struct sigmoid_forward<double, device::CPU>;

template struct sigmoid_backward<float, device::CPU>;
template struct sigmoid_backward<double, device::CPU>;

} // namespace ops