/**
 * @file cross_entropy.cpp
 * @brief cross entropy operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/functions/cross_entropy.h"

#include "error/error.h"

namespace ops {

template <typename Tp>
struct cross_entropy_forward<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,         // (scalar)
        const Tp* input,    // (batch_size, num_classes)
        const int* target,   // (batch_size)
        size_t batch_size,
        size_t num_classes
    ) {
        *output = static_cast<Tp>(0);
        for (int i = 0; i < batch_size; ++i) {
            *output += -log(input[i * num_classes + target[i]]);
        }
    }
};

#ifndef __CUDA

template <typename Tp>
struct cross_entropy_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        const Tp* input,
        const int* target,
        size_t batch_size,
        size_t num_classes
    ) {
        throw error::DeviceError("cross_entropy_forward<GPU> can not be called without CUDA support.");
    }
};

template struct cross_entropy_forward<int, device::GPU>;
template struct cross_entropy_forward<float, device::GPU>;
template struct cross_entropy_forward<double, device::GPU>;

#endif

template struct cross_entropy_forward<int, device::CPU>;
template struct cross_entropy_forward<float, device::CPU>;
template struct cross_entropy_forward<double, device::CPU>;
    
} // namespace op
