#ifndef CSRC_MACROS_H
#define CSRC_MACROS_H

// Define the number of threads per block
#define CUDA_K_THREADS 512

// Calculate the number of blocks
#define CUDA_GET_BLOCKS(N) ((N + CUDA_K_THREADS - 1) / CUDA_K_THREADS)

//Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n);                                            \
    i += blockDim.x * gridDim.x)                        

#endif