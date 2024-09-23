#ifndef CSRC_MACROS_H
#define CSRC_MACROS_H

// Define the number of threads per block
#define K_CUDA_THREADS 512

// Calculate the number of blocks
#define CUDA_GET_BLOCKS(N) ((N + K_CUDA_THREADS - 1) / K_CUDA_THREADS)

//Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n);                                            \
    i += blockDim.x * gridDim.x)                        

#endif