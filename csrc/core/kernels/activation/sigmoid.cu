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

template <typename Tp>
__global__ void 
kernel_sigmoid_f(Tp* output, Tp* input, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
}

template <typename Tp>
__global__ void 
kernel_sigmoid_b(Tp* output, Tp* input, Tp* grad, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        Tp sigmoid = 1 / (1 + exp(-input[i]));
        output[i] = sigmoid * (1 - sigmoid) * grad[i];
    }
}

template <typename Tp>
void sigmoid_forward(device::GPU* device, Tp* output, Tp* input, size_t size) {
    kernel_sigmoid_f<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, size);
}

template <typename Tp>
void sigmoid_backward(device::GPU* device, Tp* output, Tp* input, Tp* grad, size_t size) {
    kernel_sigmoid_b<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, grad, size);
}

template void sigmoid_forward<float>(device::GPU* device, float* output, float* input, size_t size);
template void sigmoid_forward<double>(device::GPU* device, double* output, double* input, size_t size);

template void sigmoid_backward<float>(device::GPU* device, float* output, float* input, float* grad, size_t size);
template void sigmoid_backward<double>(device::GPU* device, double* output, double* input, double* grad, size_t size);
