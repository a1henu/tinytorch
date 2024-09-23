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
__global__ void kernel_sigmoid_f(Tp* output, Tp* input, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
}

template <typename Tp>
__global__ void kernel_sigmoid_b(Tp* output, Tp* input, Tp* grad, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        Tp sigmoid = 1 / (1 + exp(-input[i]));
        output[i] = sigmoid * (1 - sigmoid) * grad[i];
    }
}

template <typename Tp>
void sigmoid_forward(Tp* output, Tp* input, size_t size) {
    kernel_sigmoid_f<Tp><<<CUDA_GET_BLOCKS(size), K_CUDA_THREADS>>>(output, input, size);
}

template <typename Tp>
void sigmoid_backward(Tp* output, Tp* input, Tp* grad, size_t size) {
    kernel_sigmoid_b<Tp><<<CUDA_GET_BLOCKS(size), K_CUDA_THREADS>>>(output, input, grad, size);
}