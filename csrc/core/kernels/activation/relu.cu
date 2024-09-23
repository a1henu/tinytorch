/**
 * @file relu.cu
 * @brief relu operator implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"
#include "core/kernels/activation/relu.h"
#include "macros.h"

template <typename Tp>
__global__ void 
kernel_relu_f(Tp* output, Tp* input, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

template <typename Tp>
__global__ void 
kernel_relu_b(Tp* output, Tp* input, Tp* grad, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input[i] > 0 ? grad[i] : 0;
    }
}


template <typename Tp>
void relu_forward(device::GPU* device, Tp* output, Tp* input, size_t size) {
    kernel_relu_f<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, size);
}

template <typename Tp>
void relu_backward(device::GPU* device, Tp* output, Tp* input, Tp* grad, size_t size) {
    kernel_relu_b<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, grad, size);
}

template void relu_forward<float>(device::GPU* device, float* output, float* input, size_t size);
template void relu_forward<double>(device::GPU* device, double* output, double* input, size_t size);

template void relu_backward<float>(device::GPU* device, float* output, float* input, float* grad, size_t size);
template void relu_backward<double>(device::GPU* device, double* output, double* input, double* grad, size_t size);
