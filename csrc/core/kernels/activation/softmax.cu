/**
 * @file softmax.cpp
 * @brief softmax operator implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/activation/softmax.h"

namespace ops {

template <typename Tp>
__global__ void
kernel_softmax(
    Tp* output,
    const Tp* input,
    size_t batch_size,
    size_t num_classes
) {

}

template <typename Tp>
struct softmax_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        const Tp* input,
        size_t batch_size,
        size_t num_classes
    ) {
        
    }
};


}