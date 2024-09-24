/**
 * @file ops.cpp
 * @brief Math operators implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"
#include "core/kernels/ops.h"

#include "error/error.h"

namespace ops {

template <typename Tp>
struct add_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = input1[i] + input2[i];
        }
    }
};

template <typename Tp>
struct sub_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = input1[i] - input2[i];
        }
    }
};

template <typename Tp>
struct equal_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        bool* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        *output = true;
        for (int i = 0; i < size; ++i) {
            if (input1[i] != input2[i]) {
                *output = false;
            }
        }
    }
};

#ifndef __CUDA

template <typename Tp>
struct add_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        throw error::DeviceError("add_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct sub_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        throw error::DeviceError("sub_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct equal_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        bool* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        throw error::DeviceError("equal_op<GPU> can not be called without CUDA support.");
    }
};

template struct add_op<int, device::GPU>;
template struct add_op<float, device::GPU>;
template struct add_op<double, device::GPU>;

template struct sub_op<int, device::GPU>;
template struct sub_op<float, device::GPU>;
template struct sub_op<double, device::GPU>;

template struct equal_op<int, device::GPU>;
template struct equal_op<float, device::GPU>;
template struct equal_op<double, device::GPU>;

#endif

template struct add_op<int, device::CPU>;
template struct add_op<float, device::CPU>;
template struct add_op<double, device::CPU>;

template struct sub_op<int, device::CPU>;
template struct sub_op<float, device::CPU>;
template struct sub_op<double, device::CPU>;

template struct equal_op<int, device::CPU>;
template struct equal_op<float, device::CPU>;
template struct equal_op<double, device::CPU>;

} // namespace ops