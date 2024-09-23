/**
 * @file relu.cu
 * @brief relu operator implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

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
void relu_forward(Tp* output, Tp* input, size_t size) {
    kernel_relu_f<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, size);
}

template <typename Tp>
void relu_backward(Tp* output, Tp* input, Tp* grad, size_t size) {
    kernel_relu_b<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input, grad, size);
}