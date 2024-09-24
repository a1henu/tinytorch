/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 */

#include <vector>

#include "core/device/device.h"
#include "core/memory/memory.h"
#include "error/error.h"

#include "tensor.h"

namespace tensor {

template<typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    device::BaseDevice* device
) : shape(shape), device(device) {
    size_t tol_size = get_tol_size();
    if (device->is_cpu() || device->is_gpu()) {
        memory::malloc_mem_op(device, p_data, tol_size);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template<typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    device::BaseDevice* device,
    const std::vector<Tp>& vec
) : shape(shape), device(device) {
    size_t tol_size = get_tol_size();
    if (device->is_cpu() || device->is_gpu()) {
        memory::malloc_mem_op(device, p_data, tol_size);
        memory::copy_mem_op(device, device::cpu_device, p_data, vec.data(), std::min(tol_size, vec.size()));
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template<typename Tp>
Tensor<Tp>::Tensor(Tensor<Tp>&& other) : shape(std::move(other.shape)) {
    device = other.device;
    p_data = other.p_data;
    other.p_data = nullptr;
}

template<typename Tp>
Tensor<Tp>::~Tensor() {
    if (device->is_cpu() || device->is_gpu()) {
        memory::free_mem_op(device, p_data);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template<typename Tp>
inline Tensor<Tp>& Tensor<Tp>::cpu() {
    if (device->is_cpu()) {
        return *this;
    } else if (device->is_gpu()) {
        size_t tol_size = get_tol_size();
        Tp* tmp_data;

        memory::malloc_mem_op(device::cpu_device, tmp_data, tol_size);
        memory::copy_mem_op(device::cpu_device, device, tmp_data, p_data, tol_size);
        memory::free_mem_op(device, p_data);
        memory::malloc_mem_op(device::cpu_device, p_data, tol_size);
        memory::copy_mem_op(device::cpu_device, device::cpu_device, p_data, tmp_data, tol_size);
        memory::free_mem_op(device::cpu_device, tmp_data);
        
        device = device::cpu_device;
        return *this;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template<typename Tp>
inline Tensor<Tp>& Tensor<Tp>::gpu() {
    if (device->is_gpu()) {
        return *this;
    } else if (device->is_cpu()) {
        size_t tol_size = get_tol_size();
        Tp* tmp_data;

        memory::malloc_mem_op(device::gpu_device, tmp_data, tol_size);
        memory::copy_mem_op(device::gpu_device, device, tmp_data, p_data, tol_size);
        memory::free_mem_op(device, p_data);
        memory::malloc_mem_op(device::gpu_device, p_data, tol_size);
        memory::copy_mem_op(device::gpu_device, device::gpu_device, p_data, tmp_data, tol_size);
        memory::free_mem_op(device::gpu_device, tmp_data);
        
        device = device::gpu_device;
        return *this;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}


template<typename Tp>
size_t Tensor<Tp>::get_tol_size() const {
    size_t s = 1;
    for (auto& i : shape) {
        s *= i;
    }
    return s;
}

}