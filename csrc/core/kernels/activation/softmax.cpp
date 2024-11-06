/**
 * @file softmax.cpp
 * @brief softmax operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/activation/softmax.h"

#include "error/error.h"

namespace ops { 

template <typename Tp>
struct softmax_forward<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,
        const Tp* input,
        size_t batch_size,
        size_t num_classes
    ) {
        for (int i = 0; i < batch_size; ++i) {
            Tp max_val = input[i * num_classes];
            for (int j = 1; j < num_classes; ++j) {
                max_val = std::max(max_val, input[i * num_classes + j]);
            }

            Tp sum = 0;
            for (int j = 0; j < num_classes; ++j) {
                output[i * num_classes + j] = exp(input[i * num_classes + j] - max_val);
                sum += output[i * num_classes + j];
            }

            for (int j = 0; j < num_classes; ++j) {
                output[i * num_classes + j] /= sum;
            }
        }
    }
};

#ifndef __CUDA

template <typename Tp>
struct softmax_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        const Tp* input,
        size_t batch_size,
        size_t num_classes
    ) {
        throw error::DeviceError("softmax_forward<GPU> can not be called without CUDA support.");
    }
};

template struct softmax_forward<int, device::GPU>;
template struct softmax_forward<float, device::GPU>;
template struct softmax_forward<double, device::GPU>;

#endif

template struct softmax_forward<int, device::CPU>;
template struct softmax_forward<float, device::CPU>;
template struct softmax_forward<double, device::CPU>;

} // namespace ops