/**
 * @file memory.cu
 * @brief Memory operator implementations for GPU.
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"
#include "memory.h"

namespace memory {

template <typename Tp>
struct malloc_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data, const size_t size) {
        cudaMalloc(&p_data, size * sizeof(Tp));
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data) {
        cudaFree(p_data);
    }
};

template <typename Tp>
struct copy_mem_op<Tp, device::CPU, device::GPU> {
    void operator()(
        const device::CPU* dev_dst, 
        const device::GPU* dev_src, 
        Tp* p_dst, 
        const Tp* p_src, 
        const size_t size
    ) {
        cudaMemcpy(p_dst, p_src, size * sizeof(Tp), cudaMemcpyDeviceToHost);
    }
};

template <typename Tp>
struct copy_mem_op<Tp, device::GPU, device::CPU> {
    void operator()(
        const device::GPU* dev_dst, 
        const device::CPU* dev_src, 
        Tp* p_dst, 
        const Tp* p_src, 
        const size_t size
    ) {
        cudaMemcpy(p_dst, p_src, size * sizeof(Tp), cudaMemcpyHostToDevice);
    }
};

template <typename Tp>
struct copy_mem_op<Tp, device::GPU, device::GPU> {
    void operator()(
        const device::GPU* dev_dst, 
        const device::GPU* dev_src, 
        Tp* p_dst, 
        const Tp* p_src, 
        const size_t size
    ) {
        cudaMemcpy(p_dst, p_src, size * sizeof(Tp), cudaMemcpyDeviceToDevice);
    }
};

template <typename Tp>
struct set_mem_op<Tp, device::GPU> {
    void operator()(
        const device::GPU* device, 
        Tp* p_data, 
        const Tp value, 
        const size_t size
    ) {
        cudaMemset(p_data, value, size * sizeof(Tp));
    }
};

template struct malloc_mem_op<int, device::GPU>;
template struct malloc_mem_op<float, device::GPU>;
template struct malloc_mem_op<double, device::GPU>;
template struct malloc_mem_op<bool, device::GPU>;

template struct free_mem_op<int, device::GPU>;
template struct free_mem_op<float, device::GPU>;
template struct free_mem_op<double, device::GPU>;
template struct free_mem_op<bool, device::GPU>;

template struct copy_mem_op<int, device::GPU, device::CPU>;
template struct copy_mem_op<float, device::GPU, device::CPU>;
template struct copy_mem_op<double, device::GPU, device::CPU>;
template struct copy_mem_op<bool, device::GPU, device::CPU>;
template struct copy_mem_op<int, device::CPU, device::GPU>;
template struct copy_mem_op<float, device::CPU, device::GPU>;
template struct copy_mem_op<double, device::CPU, device::GPU>;
template struct copy_mem_op<bool, device::CPU, device::GPU>;
template struct copy_mem_op<int, device::GPU, device::GPU>;
template struct copy_mem_op<float, device::GPU, device::GPU>;
template struct copy_mem_op<double, device::GPU, device::GPU>;
template struct copy_mem_op<bool, device::GPU, device::GPU>;

template struct set_mem_op<int, device::GPU>;
template struct set_mem_op<float, device::GPU>;
template struct set_mem_op<double, device::GPU>;
template struct set_mem_op<bool, device::GPU>;

} // namespace memory
