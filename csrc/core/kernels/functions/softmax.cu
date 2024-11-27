/**
 * @file softmax.cu
 * @brief softmax operator implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/functions/softmax.h"

namespace ops {

template <typename Tp>
__global__ void
kernel_max(
    Tp* output,
    const Tp* input,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size) {
        Tp max_val = input[i * num_classes];
        for (size_t j = 1; j < num_classes; ++j) {
            max_val = max(max_val, input[i * num_classes + j]);
        }
        output[i] = max_val;
    }
}

template <typename Tp>
__global__ void
kernel_sub(
    Tp* output,
    const Tp* input,
    const Tp* max_val,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(k, batch_size * num_classes) {
        size_t i = k / num_classes;
        size_t j = k % num_classes;
        output[i * num_classes + j] = input[i * num_classes + j] - max_val[i];
    }
}

template <typename Tp>
__global__ void
kernel_exp(
    Tp* output,
    const Tp* input,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(k, batch_size * num_classes) {
        output[k] = expf(input[k]);
    }
}

template <typename Tp>
__global__ void
kernel_sum(
    Tp* output,
    const Tp* input,
    Tp* sum_val,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size) {
        Tp sum = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            sum += input[i * num_classes + j];
        }
        sum_val[i] = sum;
    }
}

template <typename Tp>
__global__ void
kernel_div(
    Tp* output,
    const Tp* input,
    const Tp* sum_val,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(k, batch_size * num_classes) {
        size_t i = k / num_classes;
        size_t j = k % num_classes;
        output[i * num_classes + j] = input[i * num_classes + j] / sum_val[i];
    }
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
        Tp* max_val;
        Tp* sum_val;
        cudaMalloc(&max_val, batch_size * sizeof(Tp));
        cudaMalloc(&sum_val, batch_size * sizeof(Tp));

        kernel_max<Tp><<<CUDA_GET_BLOCKS(batch_size), CUDA_K_THREADS>>>(
            max_val, input, batch_size, num_classes
        );
        cudaDeviceSynchronize();
        kernel_sub<Tp><<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            output, input, max_val, batch_size, num_classes
        );
        cudaDeviceSynchronize();
        kernel_exp<Tp><<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            output, output, batch_size, num_classes
        );
        cudaDeviceSynchronize();
        kernel_sum<Tp><<<CUDA_GET_BLOCKS(batch_size), CUDA_K_THREADS>>>(
            sum_val, output, sum_val, batch_size, num_classes
        );
        cudaDeviceSynchronize();
        kernel_div<Tp><<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            output, output, sum_val, batch_size, num_classes
        );
        cudaDeviceSynchronize();
    }
};

template struct softmax_forward<int, device::GPU>;
template struct softmax_forward<float, device::GPU>;
template struct softmax_forward<double, device::GPU>;

} //  namespace ops