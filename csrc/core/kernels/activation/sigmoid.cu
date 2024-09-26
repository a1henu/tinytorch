/**
 * @file sigmoid.cu
 * @brief sigmoid operator implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/activation/sigmoid.h"
#include "macros.h"

namespace ops {

template <typename Tp>
__global__ void 
kernel_sigmoid_f(Tp* output, const Tp* input, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = 1 / (1 + expf(-input[i]));
    }
}

template <typename Tp>
__global__ void 
kernel_sigmoid_b(Tp* output, const Tp* input, const Tp* grad, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        Tp sigmoid = 1 / (1 + expf(-input[i]));
        output[i] = sigmoid * (1 - sigmoid) * grad[i];
    }
}

template <typename Tp>
struct sigmoid_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input, 
        size_t size
    ) {
        kernel_sigmoid_f<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, size);
    }
};

template <typename Tp>
struct sigmoid_backward<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input, 
        const Tp* grad, 
        size_t size
    ) {
        kernel_sigmoid_b<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, grad, size);
    }
};

template struct sigmoid_forward<int, device::GPU>;
template struct sigmoid_forward<float, device::GPU>;
template struct sigmoid_forward<double, device::GPU>;

template struct sigmoid_backward<int, device::GPU>;
template struct sigmoid_backward<float, device::GPU>;
template struct sigmoid_backward<double, device::GPU>;

} // namespace ops