/**
 * @file macros.h
 * @brief Macros for tinytorch
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_MACROS_H
#define CSRC_MACROS_H

#include "core/device/device.h"

// Define the number of threads per block
#define CUDA_K_THREADS 512

// Calculate the number of blocks
#define CUDA_GET_BLOCKS(N) ((N + CUDA_K_THREADS - 1) / CUDA_K_THREADS)

//Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n);                                            \
    i += blockDim.x * gridDim.x)                        


// Define the type getter of the tensor
template <typename Tp>
struct get_device_type { };

template <>
struct get_device_type<device::CPU> {
    using type = device::CPU;
};

template <>
struct get_device_type<device::GPU> {
    using type = device::GPU;
};

#endif
