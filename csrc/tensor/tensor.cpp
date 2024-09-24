/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
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
    if (device->is_cpu()) {
        malloc_cpu_op()()(device, p_data, tol_size);
    } else if (device->is_gpu()) {
        malloc_gpu_op()()(device, p_data, tol_size);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template<typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    Tp* data, 
    device::BaseDevice* t_device, 
    device::BaseDevice* d_device
) : shape(shape), device(t_device) {
    size_t tol_size = get_tol_size();
    if (t_device->is_cpu()) {
        malloc_cpu_op()(t_device, p_data, tol_size);
        if (d_device->is_cpu()) {
            copy_c2c_op()(t_device, d_device, p_data, data, tol_size);
        } else if (d_device->is_gpu()) {
            copy_g2c_op()(t_device, d_device, p_data, data, tol_size);
        } else {
            throw error::DeviceError("Unknown device type");
        }
    } else if (t_device->is_gpu()) {
        malloc_gpu_op()(t_device, p_data, tol_size);
        if (d_device->is_cpu()) {
            copy_c2g_op()(t_device, d_device, p_data, data, tol_size);
        } else if (d_device->is_gpu()) {
            copy_g2g_op()(device::gpu_device, device, p_data, data, tol_size);
        } else {
            throw error::DeviceError("Unknown device type");
        }
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template<typename Tp>
Tensor<Tp>::Tensor(const Tensor<Tp>& other) : shape(other.shape), device(other.device) {
    size_t tol_size = get_tol_size();
    if (device->is_cpu()) {
        malloc_cpu_op()(device, p_data, tol_size);
        copy_c2c_op()(device, other.device, p_data, other.p_data, tol_size);
    } else if (device->is_gpu()) {
        malloc_gpu_op()(device, p_data, tol_size);
        copy_g2g_op()(device, other.device, p_data, other.p_data, tol_size);
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
    if (device->is_cpu()) {
        free_cpu_op()(device, p_data);
    } else if (device->is_gpu()) {
        free_gpu_op()(device, p_data);
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

        malloc_cpu_op()(device::cpu_device, tmp_data, tol_size);
        copy_g2c_op()(device::cpu_device, device, tmp_data, p_data, tol_size);
        free_gpu_op()(device, p_data);
        malloc_cpu_op()(device::cpu_device, p_data, tol_size);
        copy_c2c_op()(device::cpu_device, device::cpu_device, p_data, tmp_data, tol_size);
        free_cpu_op()(device::cpu_device, tmp_data);
        
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

        malloc_gpu_op()(device::gpu_device, tmp_data, tol_size);
        copy_c2g_op()(device::gpu_device, device, tmp_data, p_data, tol_size);
        free_cpu_op()(device, p_data);
        malloc_gpu_op()(device::gpu_device, p_data, tol_size);
        copy_g2g_op()(device::gpu_device, device::gpu_device, p_data, tmp_data, tol_size);
        free_gpu_op()(device::gpu_device, tmp_data);
        
        device = device::gpu_device;
        return *this;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
bool Tensor<Tp>::in_cpu() const {
    return device->is_cpu();
}

template <typename Tp>
bool Tensor<Tp>::in_gpu() const {
    return device->is_gpu();
}

template <typename Tp>
const std::vector<int> Tensor<Tp>::get_shape() const {
    return shape;
}

template <typename Tp>
const Tp* Tensor<Tp>::get_data() const {
    return p_data;
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