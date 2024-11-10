/**
 * @file cross_entropy.cpp
 * @brief cross entropy operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/functions/softmax.h"
#include "core/kernels/functions/cross_entropy.h"

#include "error/error.h"

namespace ops {

template <typename Tp>
struct cross_entropy_forward<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,         // (scalar)
        const Tp* input,    // (batch_size, num_classes)
        const Tp* target,   // (batch_size)
        size_t batch_size,
        size_t num_classes
    ) {
        *output = static_cast<Tp>(0);
        for (int i = 0; i < batch_size; ++i) {
            *output -= log(input[i * num_classes + static_cast<int>(target[i])]);
        }
        *output /= batch_size;
    }
};

template <typename Tp>
struct cross_entropy_backward<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,         // (batch_size, num_classes)
        const Tp* input,    // (batch_size, num_classes)
        const Tp* target,  // (batch_size)
        size_t batch_size,
        size_t num_classes
    ) {
        softmax_forward<Tp, device::CPU>()(device, output, input, batch_size, num_classes);
        for (int i = 0; i < batch_size; ++i) {
            output[i * num_classes + static_cast<int>(target[i])] -= static_cast<Tp>(1);
            for (int j = 0; j < num_classes; ++j) {
                output[i * num_classes + j] /= batch_size;
            }
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
        const Tp* target,
        size_t batch_size,
        size_t num_classes
    ) {
        throw error::DeviceError("cross_entropy_forward<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct cross_entropy_backward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input, 
        const Tp* target, 
        size_t batch_size,
        size_t num_classes
    ) {
        throw error::DeviceError("cross_entropy_backward<GPU> can not be called without CUDA support.");
    }
};

template struct cross_entropy_forward<int, device::GPU>;
template struct cross_entropy_forward<float, device::GPU>;
template struct cross_entropy_forward<double, device::GPU>;

template struct cross_entropy_backward<int, device::GPU>;
template struct cross_entropy_backward<float, device::GPU>;
template struct cross_entropy_backward<double, device::GPU>;

#endif

template struct cross_entropy_forward<int, device::CPU>;
template struct cross_entropy_forward<float, device::CPU>;
template struct cross_entropy_forward<double, device::CPU>;

template struct cross_entropy_backward<int, device::CPU>;
template struct cross_entropy_backward<float, device::CPU>;
template struct cross_entropy_backward<double, device::CPU>;
    
} // namespace op
