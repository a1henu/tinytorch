/**
 * @file memory.cpp
 * @brief Memory operator implementations for CPU.
 */

#include <cstring>

#include "core/device/device.h"
#include "memory.h"

namespace memory{

template <typename Tp>
struct malloc_mem_op<Tp, device::CPU> {
    void operator()(const device::CPU* device, Tp*& p_data, const size_t size) {
        if (p_data != nullptr) {
            free(p_data);
        }
        p_data = static_cast<Tp*>(malloc(size * sizeof(Tp)));
    }
};

template <typename Tp>
struct malloc_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data, const size_t size) {
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::CPU> {
    void operator()(const device::CPU* device, Tp*& p_data) {
        free(p_data);
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data) {
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
struct copy_mem_op<Tp, device::CPU, device::GPU> {
    void operator()(
        const device::CPU* dev_dst, 
        const device::GPU* dev_src, 
        Tp* p_dst, 
        const Tp* p_src, 
        const size_t size
    ) {
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

template <typename Tp>
struct set_mem_op<Tp, device::GPU> {
    void operator()(
        const device::GPU* device, 
        Tp* p_data, 
        const Tp value, 
        const size_t size
    ) {
    }
};

} // namespace memory