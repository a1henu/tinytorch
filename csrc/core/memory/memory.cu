#include "core/device/device.h"
#include "memory.h"

namespace memory {

template <typename Tp>
struct malloc_mem_op<Tp, device::CPU> {
    void operator()(const device::CPU* device, Tp*& p_data, const size_t size) {
    }
};

template <typename Tp>
struct malloc_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data, const size_t size) {
        if (p_data != nullptr) {
            cudaFree(p_data);
        }
        cudaMalloc(&p_data, size * sizeof(Tp));
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::CPU> {
    void operator()(const device::CPU* device, Tp*& p_data) {
    }
};

template <typename Tp>
struct free_mem_op<Tp, device::GPU> {
    void operator()(const device::GPU* device, Tp*& p_data) {
        cudaFree(p_data);
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

}
