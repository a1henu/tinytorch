/**
 * @file softmax.cpp
 * @brief softmax operator implementation for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/activation/softmax.h"

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

} // namespace ops