/**
 * @file mse.cpp
 * @brief Mean Squared Error operator implementation for CPU
 * 
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/functions/mse.h"
#include "error/error.h"

namespace ops {

template <typename Tp>
struct mse_forward<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,         // (scalar)
        const Tp* input,    // (batch_size, num_classes)
        const Tp* target,   // (batch_size, num_classes)
        size_t batch_size,
        size_t num_classes
    ) {
        *output = static_cast<Tp>(0);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_classes; ++j) {
                Tp diff = input[i * num_classes + j] - target[i * num_classes + j];
                *output += diff * diff;
            }
        }
        *output /= (batch_size * num_classes);
    }
};

template <typename Tp>
struct mse_backward<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,         // (batch_size, num_classes)
        const Tp* input,    // (batch_size, num_classes)
        const Tp* target,   // (batch_size, num_classes)
        size_t batch_size,
        size_t num_classes
    ) {
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_classes; ++j) {
                output[i * num_classes + j] = 2 * (input[i * num_classes + j] - target[i * num_classes + j]) / (batch_size * num_classes);
            }
        }
    }
};

template struct mse_forward<int, device::CPU>;
template struct mse_forward<float, device::CPU>;
template struct mse_forward<double, device::CPU>;

template struct mse_backward<int, device::CPU>;
template struct mse_backward<float, device::CPU>;
template struct mse_backward<double, device::CPU>;

} // namespace ops