/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cassert>
#include <vector>

#include "core/device/device.h"
#include "core/memory/memory.h"
#include "error/error.h"

#include "tensor/tensor.h"

namespace tensor {

template <typename Tp>
Tensor<Tp>::Tensor() { }

template <typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    DeviceType device
) : shape(shape) {
    size_t tol_size = get_tol_size();
    if (device == DeviceType::CPU) {
        this->device = new device::CPU();
        malloc_cpu_op()(device::cpu_device, p_data, tol_size);
    } else if (device == DeviceType::GPU) {
        this->device = new device::GPU();
        malloc_gpu_op()(device::gpu_device, p_data, tol_size);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    DeviceType device,
    Tp* data
) : shape(shape) {
    size_t tol_size = get_tol_size();
    if (device == DeviceType::CPU) {
        this->device = new device::CPU();
        malloc_cpu_op()(device::cpu_device, p_data, tol_size);
        copy_c2c_op()(device::cpu_device, device::cpu_device, p_data, data, tol_size);
    } else if (device == DeviceType::GPU) {
        this->device = new device::GPU();
        malloc_gpu_op()(device::gpu_device, p_data, tol_size);
        copy_g2g_op()(device::gpu_device, device::gpu_device, p_data, data, tol_size);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
Tensor<Tp>::Tensor(const Tensor<Tp>& other) : shape(other.shape) {
    size_t tol_size = get_tol_size();
    if (other.in_cpu()) {
        this->device = new device::CPU();
        malloc_cpu_op()(device::cpu_device, p_data, tol_size);
        copy_c2c_op()(device::cpu_device, device::cpu_device, p_data, other.p_data, tol_size);
    } else if (other.in_gpu()) {
        this->device = new device::GPU();
        malloc_gpu_op()(device::gpu_device, p_data, tol_size);
        copy_g2g_op()(device::gpu_device, device::gpu_device, p_data, other.p_data, tol_size);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
Tensor<Tp>& Tensor<Tp>::operator=(const Tensor<Tp>& other) {
    if (this != &other) {
        if (p_data != nullptr) {
            if (device->is_cpu()) {
                free_cpu_op()(device::cpu_device, p_data);
            } else if (device->is_gpu()) {
                free_gpu_op()(device::gpu_device, p_data);
            }
        }

        shape = other.shape;
        size_t tol_size = get_tol_size();
        if (other.in_cpu()) {
            this->device = new device::CPU();
            malloc_cpu_op()(device::cpu_device, p_data, tol_size);
            copy_c2c_op()(device::cpu_device, device::cpu_device, p_data, other.p_data, tol_size);
        } else if (other.in_gpu()) {
            this->device = new device::GPU();
            malloc_gpu_op()(device::gpu_device, p_data, tol_size);
            copy_g2g_op()(device::gpu_device, device::gpu_device, p_data, other.p_data, tol_size);
        }
    }
    return *this;
}

template <typename Tp>
Tensor<Tp>::~Tensor() {
    if (p_data != nullptr) {
        if (device->is_cpu()) {
            free_cpu_op()(device::cpu_device, p_data);
        } else if (device->is_gpu()) {
            free_gpu_op()(device::gpu_device, p_data);
        }
    }
    if (device != nullptr) {
        delete device;
    }
}

template <typename Tp>
inline Tensor<Tp>& Tensor<Tp>::cpu() {
    if (device->is_cpu()) {
        return *this;
    } else if (device->is_gpu()) {
        size_t tol_size = get_tol_size();
        Tp* tmp_data;

        malloc_cpu_op()(device::cpu_device, tmp_data, tol_size);
        copy_g2c_op()(device::cpu_device, device::gpu_device, tmp_data, p_data, tol_size);
        free_gpu_op()(device::gpu_device, p_data);
        malloc_cpu_op()(device::cpu_device, p_data, tol_size);
        copy_c2c_op()(device::cpu_device, device::cpu_device, p_data, tmp_data, tol_size);
        free_cpu_op()(device::cpu_device, tmp_data);
        
        delete device;
        device = new device::CPU();

        return *this;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
inline Tensor<Tp>& Tensor<Tp>::gpu() {
    if (device->is_gpu()) {
        return *this;
    } else if (device->is_cpu()) {
        size_t tol_size = get_tol_size();
        Tp* tmp_data;

        malloc_gpu_op()(device::gpu_device, tmp_data, tol_size);
        copy_c2g_op()(device::gpu_device, device::cpu_device, tmp_data, p_data, tol_size);
        free_cpu_op()(device::cpu_device, p_data);
        malloc_gpu_op()(device::gpu_device, p_data, tol_size);
        copy_g2g_op()(device::gpu_device, device::gpu_device, p_data, tmp_data, tol_size);
        free_gpu_op()(device::gpu_device, tmp_data);
        
        delete device;
        device = new device::GPU();

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

template <typename Tp>
void Tensor<Tp>::set_data(Tp* data, size_t size, DeviceType device_d) const {
    size_t tol_size = get_tol_size();
    assert(size <= tol_size);
    if (device->is_cpu() && device_d == DeviceType::CPU) {
        if (p_data != nullptr) {
            free_cpu_op()(device::cpu_device, p_data);
        }
        copy_c2c_op()(device::cpu_device, device::cpu_device, p_data, data, tol_size);
    } else if (device->is_gpu() && device_d == DeviceType::GPU) {
        if (p_data != nullptr) {
            free_gpu_op()(device::gpu_device, p_data);
        }
        copy_g2g_op()(device::gpu_device, device::gpu_device, p_data, data, tol_size);
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
size_t Tensor<Tp>::get_tol_size() const {
    size_t s = 1;
    for (auto& i : shape) {
        s *= i;
    }
    return s;
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

}