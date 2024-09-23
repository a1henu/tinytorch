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

namespace activation {

template <typename Tp>
struct relu_forward<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        Tp* input, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }
};

template <typename Tp>
struct relu_backward<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        Tp* input, 
        Tp* grad, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = input[i] > 0 ? grad[i] : 0;
        }
    }
};

#ifndef __CUDA

template <typename Tp>
struct relu_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input, 
        size_t size
    ) {
        throw error::DeviceError("relu_forward<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct relu_backward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input, 
        Tp* grad, 
        size_t size
    ) {
        throw error::DeviceError("relu_backward<GPU> can not be called without CUDA support.");
    }
};

template struct relu_forward<float, device::GPU>;
template struct relu_forward<double, device::GPU>;

template struct relu_backward<float, device::GPU>;
template struct relu_backward<double, device::GPU>;

#endif

template struct relu_forward<float, device::CPU>;
template struct relu_forward<double, device::CPU>;

template struct relu_backward<float, device::CPU>;
template struct relu_backward<double, device::CPU>;

} // namespace activation