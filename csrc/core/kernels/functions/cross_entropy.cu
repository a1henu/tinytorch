/**
 * @file cross_entropy.cu
 * @brief cross entropy operator implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cmath>

#include "core/kernels/functions/softmax.h"
#include "core/kernels/functions/cross_entropy.h"

#include "error/error.h"
#include "macros.h"

namespace ops {

template <typename Tp>
__global__ void kernel_log(
    const Tp* input, 
    Tp* output, 
    const int* target, 
    size_t batch_size, 
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size) {
        output[i] = logf(input[i * num_classes + target[i]]);
    }
}

template <typename Tp>
__global__ void 
kernel_neg_sum(
    const Tp* input, 
    Tp* output, 
    size_t batch_size
) {
    for (size_t i = 0; i < batch_size; i++) {
        *output -= input[i];
    }
    *output /= static_cast<Tp>(batch_size);
}

template <typename Tp>
__global__ void
kernel_set_zero(
    Tp* input, 
    size_t size
) {
    CUDA_KERNEL_LOOP(i, size) {
        input[i] = 0;
    }
}

template <typename Tp>
__global__ void
kernel_sub_target(
    Tp* arr,
    const int* target,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size) {
        arr[i * num_classes + target[i]] -= static_cast<Tp>(1);
    }
}

template <typename Tp>
__global__ void
kernel_div_batch_size(
    Tp* arr,
    size_t batch_size,
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size * num_classes) {
        arr[i] /= static_cast<Tp>(batch_size);
    }
}

template <typename Tp>
struct cross_entropy_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,          // (scalar)
        const Tp* input,     // (batch_size, num_classes)
        const int* target,   // (batch_size)
        size_t batch_size,
        size_t num_classes
    ) {
        kernel_set_zero<Tp><<<1, 1>>>(output, 1);

        Tp* log_input;
        cudaMalloc(&log_input, batch_size * sizeof(Tp));

        kernel_log<Tp><<<CUDA_GET_BLOCKS(batch_size), CUDA_K_THREADS>>>(
            input, log_input, target, batch_size, num_classes
        );
        cudaDeviceSynchronize();
        
        kernel_neg_sum<Tp><<<1, 1>>>(
            log_input, output, batch_size
        );
        cudaDeviceSynchronize();

        cudaFree(log_input);
    }
};

template <typename Tp>
struct cross_entropy_backward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,          // (batch_size, num_classes)
        const Tp* input,     // (batch_size, num_classes)
        const int* target,   // (batch_size)
        size_t batch_size,
        size_t num_classes
    ) {
        softmax_forward<Tp, device::GPU>()(device, output, input, batch_size, num_classes);

        kernel_sub_target<Tp><<<CUDA_GET_BLOCKS(batch_size), CUDA_K_THREADS>>>(
            output, target, batch_size, num_classes
        );

        kernel_div_batch_size<Tp><<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            output, batch_size, num_classes
        );
    }
};

template struct cross_entropy_forward<int, device::GPU>;
template struct cross_entropy_forward<float, device::GPU>;
template struct cross_entropy_forward<double, device::GPU>;

template struct cross_entropy_backward<int, device::GPU>;
template struct cross_entropy_backward<float, device::GPU>;
template struct cross_entropy_backward<double, device::GPU>;

} // namespace ops