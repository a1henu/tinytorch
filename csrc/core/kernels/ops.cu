/**
 * @file ops.cu
 * @brief Math operators implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/kernels/ops.h"

namespace ops {

template <typename Tp>
__global__ void
kernel_add(Tp* output, Tp* input1, Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input1[i] + input2[i];
    }
}

template <typename Tp>
__global__ void
kernel_sub(Tp* output, Tp* input1, Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input1[i] - input2[i];
    }
}

template <typename Tp>
__global__ void
kernel_eq(bool* output, Tp* input1, Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        if (input1[i] != input2[i]) {
            *output = false;
        }
    }
}

template <typename Tp>
struct add_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        kernel_add<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input1, input2, size);
    }
};

template <typename Tp>
struct sub_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        kernel_sub<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input1, input2, size);
    }
};

template <typename Tp>
struct equal_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        bool* output, 
        Tp* input1, 
        Tp* input2, 
        size_t size
    ) {
        bool flag = true;
        kernel_eq<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(&flag, input1, input2, size);
        cudaDeviceSynchronize();

        *output = flag;
    }
};

template struct add_op<int, device::GPU>;
template struct add_op<float, device::GPU>;
template struct add_op<double, device::GPU>;

template struct sub_op<int, device::GPU>;
template struct sub_op<float, device::GPU>;
template struct sub_op<double, device::GPU>;

} // namespace ops