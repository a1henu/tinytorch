/**
 * @file memory.cpp
 * @brief Memory operator implementations for CPU.
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cstring>

#include "core/device/device.h"
#include "core/memory/memory.h"

#include "error/error.h"

namespace memory{

template <typename Tp>
struct malloc_mem_op<Tp, device::CPU> {
    void operator()(const device::CPU* device, Tp*& p_data, const size_t size) {
        p_data = static_cast<Tp*>(malloc(size * sizeof(Tp)));
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::CPU> {
    void operator()(const device::CPU* device, Tp*& p_data) {
        free(p_data);
    }
};

template <typename Tp>
struct copy_mem_op<Tp, device::CPU, device::CPU> {
    void operator()(
        const device::CPU* dev_dst, 
        const device::CPU* dev_src, 
        Tp* p_dst, 
        const Tp* p_src, 
        const size_t size
    ) {
        memcpy(p_dst, p_src, size * sizeof(Tp));
    }
};

template <typename Tp>
struct set_mem_op<Tp, device::CPU> {
    void operator()(
        const device::CPU* device, 
        Tp* p_data, 
        const Tp value, 
        const size_t size
    ) {
        memset(p_data, value, size * sizeof(Tp));
    }
};

#ifndef __CUDA

template <typename Tp>
struct malloc_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data, const size_t size) {
        throw error::DeviceError("malloc_mem_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data) {
        throw error::DeviceError("free_mem_op<GPU> can not be called without CUDA support.");
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
        throw error::DeviceError("copy_mem_op<GPU, CPU> can not be called without CUDA support.");
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
        throw error::DeviceError("copy_mem_op<CPU, GPU> can not be called without CUDA support.");
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
        throw error::DeviceError("copy_mem_op<GPU, GPU> can not be called without CUDA support.");
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
        throw error::DeviceError("set_mem_op<GPU> can not be called without CUDA support.");
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

#endif

template struct malloc_mem_op<int, device::CPU>;
template struct malloc_mem_op<float, device::CPU>;
template struct malloc_mem_op<double, device::CPU>;
template struct malloc_mem_op<bool, device::CPU>;

template struct free_mem_op<int, device::CPU>;
template struct free_mem_op<float, device::CPU>;
template struct free_mem_op<double, device::CPU>;
template struct free_mem_op<bool, device::CPU>;

template struct copy_mem_op<int, device::CPU, device::CPU>;
template struct copy_mem_op<float, device::CPU, device::CPU>;
template struct copy_mem_op<double, device::CPU, device::CPU>;
template struct copy_mem_op<bool, device::CPU, device::CPU>;

template struct set_mem_op<int, device::CPU>;
template struct set_mem_op<float, device::CPU>;
template struct set_mem_op<double, device::CPU>;
template struct set_mem_op<bool, device::CPU>;

} // namespace memory