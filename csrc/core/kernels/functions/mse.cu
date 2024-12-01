/**
 * @file mse.cu
 * @brief Mean Squared Error operator implementation for GPU
 * 
 * Licensed under the MIT License.
 */

#include <cmath>
#include <cuda_runtime.h>

#include "core/kernels/functions/mse.h"
#include "error/error.h"
#include "macros.h"

namespace ops {

template <typename Tp>
__global__ void
kernel_square_sub(
    const Tp* input, 
    const Tp* target, 
    Tp* output, 
    size_t batch_size, 
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size * num_classes) {
        output[i] = powf(input[i] - target[i], 2);
    }
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
kernel_mse_sum(
    const Tp* input, 
    Tp* output, 
    size_t batch_size, 
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size * num_classes) {
        *output += input[i];
    }
}

template <typename Tp>
__global__ void
kernel_mse_div(
    Tp* input, 
    size_t n
) {
    *input /= static_cast<Tp>(n);
}

template <typename Tp>
__global__ void 
kernel_mse_backward(
    const Tp* input, 
    const Tp* target, 
    Tp* output, 
    size_t batch_size, 
    size_t num_classes
) {
    CUDA_KERNEL_LOOP(i, batch_size * num_classes) {
        output[i] = 2 * (input[i] - target[i]) / (batch_size * num_classes);
    }
}

template <typename Tp>
struct mse_forward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,          // (scalar)
        const Tp* input,     // (batch_size, num_classes)
        const Tp* target,    // (batch_size, num_classes)
        size_t batch_size,
        size_t num_classes
    ) {
        Tp* square_diff;
        cudaMalloc(&square_diff, batch_size * num_classes * sizeof(Tp));
        
        kernel_square_sub<<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            input, target, square_diff, batch_size, num_classes
        );
        cudaDeviceSynchronize();

        kernel_set_zero<<<1, 1>>>(output, 1);
        kernel_mse_sum<<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            square_diff, output, batch_size, num_classes
        );
        cudaDeviceSynchronize();

        kernel_mse_div<<<1, 1>>>(output, batch_size * num_classes);
    }
};

template <typename Tp>
struct mse_backward<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,          // (batch_size, num_classes)
        const Tp* input,     // (batch_size, num_classes)
        const Tp* target,    // (batch_size, num_classes)
        size_t batch_size,
        size_t num_classes
    ) {
        kernel_mse_backward<<<CUDA_GET_BLOCKS(batch_size * num_classes), CUDA_K_THREADS>>>(
            input, target, output, batch_size, num_classes
        );
    }
};

template struct mse_forward<int, device::GPU>;
template struct mse_forward<float, device::GPU>;
template struct mse_forward<double, device::GPU>;

template struct mse_backward<int, device::GPU>;
template struct mse_backward<float, device::GPU>;
template struct mse_backward<double, device::GPU>;

} // namespace ops