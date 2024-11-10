/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cassert>
#include <iostream>

#include "core/device/device.h"
#include "core/memory/memory.h"
#include "core/kernels/ops.h"
#include "error/error.h"

#include "tensor/tensor.h"

namespace tensor {

template <typename Tp>
Tensor<Tp>::Tensor() { }

template <typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    DeviceType device
) : _shape(shape) {
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
    const Tp* data
) : _shape(shape) {
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
Tensor<Tp>::Tensor(
    const std::vector<int>& shape,
    DeviceType device,
    const std::vector<Tp>& data
): Tensor(shape, device, data.data()) { }

template <typename Tp>
Tensor<Tp>::Tensor(
    const std::initializer_list<int>& shape,
    DeviceType device
): Tensor(std::vector<int>(shape), device) { }

template <typename Tp>
Tensor<Tp>::Tensor(
    const std::initializer_list<int>& shape,
    DeviceType device,
    const Tp* data
): Tensor(std::vector<int>(shape), device, data) { }

template <typename Tp>
Tensor<Tp>::Tensor(const Tensor<Tp>& other) : _shape(other._shape) {
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

        _shape = other._shape;
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
inline Tensor<Tp> Tensor<Tp>::cpu() const {
    tensor::Tensor<Tp> out(_shape, DeviceType::CPU);
    size_t tol_size = get_tol_size();

    if (device->is_cpu()) {
        copy_c2c_op()(device::cpu_device, device::cpu_device, out.get_data(), p_data, tol_size);

        return out;    
    } else if (device->is_gpu()) {
        copy_g2c_op()(device::cpu_device, device::gpu_device, out.get_data(), p_data, tol_size);

        return out;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
inline Tensor<Tp> Tensor<Tp>::gpu() const {
    tensor::Tensor<Tp> out(_shape, DeviceType::GPU);
    size_t tol_size = get_tol_size();

    if (device->is_gpu()) {
        copy_g2g_op()(device::gpu_device, device::gpu_device, out.get_data(), p_data, tol_size);

        return out;
    } else if (device->is_cpu()) {
        copy_c2g_op()(device::gpu_device, device::cpu_device, out.get_data(), p_data, tol_size);

        return out;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
void Tensor<Tp>::to_cpu() {
    if (device->is_cpu()) {
        return;
    } else if (device->is_gpu()) {
        Tp* tmp_data;
        size_t tol_size = get_tol_size();
        memory::malloc_mem_op<Tp, device::CPU>()(device::cpu_device, tmp_data, tol_size);
        copy_g2c_op()(device::cpu_device, device::gpu_device, tmp_data, p_data, tol_size);

        if (p_data != nullptr) {
            free_gpu_op()(device::gpu_device, p_data);
        }
        p_data = tmp_data;
        device = new device::CPU();
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
void Tensor<Tp>::to_gpu() {
    if (device->is_gpu()) {
        return;
    } else if (device->is_cpu()) {
        Tp* tmp_data;
        size_t tol_size = get_tol_size();
        memory::malloc_mem_op<Tp, device::GPU>()(device::gpu_device, tmp_data, tol_size);
        copy_c2g_op()(device::gpu_device, device::cpu_device, tmp_data, p_data, tol_size);

        if (p_data != nullptr) {
            free_cpu_op()(device::cpu_device, p_data);
        }
        p_data = tmp_data;
        device = new device::GPU();
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
const int Tensor<Tp>::dim() const {
    return _shape.size();
}

template <typename Tp>
const std::vector<int> Tensor<Tp>::get_shape() const {
    return _shape;
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::reshape(const std::vector<int>& shape) const {
    int total_size = this->get_tol_size();
    int new_total_size = 1;
    int unknown_dim = -1;

    std::vector<int> new_shape(shape);

    // If the new shape has a dimension of -1, the size of the new shape must be divisible by the known dimensions.
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (unknown_dim != -1) {
                throw error::InvalidArgumentError("Only one dimension can be -1.");
            }
            unknown_dim = i;
        } else {
            new_total_size *= shape[i];
        }
    }

    if (unknown_dim != -1) {
        if (total_size % new_total_size != 0) {
            throw error::InvalidArgumentError("The size of the new shape must be divisible by the known dimensions.");
        }
        new_shape[unknown_dim] = total_size / new_total_size;
    } else if (new_total_size != total_size) {
        throw error::InvalidArgumentError("The size of the new shape must be equal to the old shape.");
    }
    if (this->in_cpu()) {
        Tensor<Tp> out(new_shape, DeviceType::CPU, this->p_data);
        return out;
    } else {
        Tensor<Tp> out(new_shape, DeviceType::GPU, this->p_data);
        return out;
    }
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::reshape(const std::initializer_list<int>& shape) const {
    return this->reshape(std::vector<int>(shape));
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::transpose() const {
    if (_shape.size() != 2) {
        throw error::InvalidArgumentError("The shape of the tensor must be 2.");
    }
    std::vector<int> new_shape {_shape[1], _shape[0]};
    if (this->in_cpu()) {
        tensor::Tensor<Tp> out(new_shape, DeviceType::CPU);
        ops::transpose_op<Tp, device::CPU>()(device::cpu_device, this->get_data(), out.get_data(), _shape[0], _shape[1]);
        return out;
    } else if (this->in_gpu()) {
        tensor::Tensor<Tp> out(new_shape, DeviceType::GPU);
        ops::transpose_op<Tp, device::GPU>()(device::gpu_device, this->get_data(), out.get_data(), _shape[0], _shape[1]);
        return out;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
Tp* Tensor<Tp>::get_data() const {
    return p_data;
}

template <typename Tp>
void Tensor<Tp>::set_data(const Tp* data, size_t size, DeviceType device_d) const {
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
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<int>());
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::operator+(const Tensor<Tp>& other) const {
    if (this->get_shape() != other.get_shape()) {
        throw error::InvalidArgumentError("The shape of two tensors must be the same.");
    }

    if (this->in_cpu() && other.in_cpu()) {
        Tensor<Tp> out(this->get_shape(), DeviceType::CPU);
        size_t tol_size = this->get_tol_size();
        ops::add_op<Tp, device::CPU>()(device::cpu_device, out.get_data(), this->get_data(), other.get_data(), tol_size);
        return out;
    } else if (this->in_gpu() && other.in_gpu()) {
        Tensor<Tp> out(this->get_shape(), DeviceType::GPU);
        size_t tol_size = this->get_tol_size();
        ops::add_op<Tp, device::GPU>()(device::gpu_device, out.get_data(), this->get_data(), other.get_data(), tol_size);
        return out;
    } else {
        throw error::DeviceError("The device of two tensors must be the same.");
    }
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::operator-(const Tensor<Tp>& other) const {
    if (this->get_shape() != other.get_shape()) {
        throw error::InvalidArgumentError("The shape of two tensors must be the same.");
    }

    if (this->in_cpu() && other.in_cpu()) {
        Tensor<Tp> out(this->get_shape(), DeviceType::CPU);
        size_t tol_size = this->get_tol_size();
        ops::sub_op<Tp, device::CPU>()(device::cpu_device, out.get_data(), this->get_data(), other.get_data(), tol_size);
        return out;
    } else if (this->in_gpu() && other.in_gpu()) {
        Tensor<Tp> out(this->get_shape(), DeviceType::GPU);
        size_t tol_size = this->get_tol_size();
        ops::sub_op<Tp, device::GPU>()(device::gpu_device, out.get_data(), this->get_data(), other.get_data(), tol_size);
        return out;
    } else {
        throw error::DeviceError("The device of two tensors must be the same.");
    }
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::operator*(const Tensor<Tp>& other) const {
    int m = this->get_shape()[0];
    int n = other.get_shape()[1];
    int k_1 = this->get_shape()[1];
    int k_2 = other.get_shape()[0];

    if (k_1 != k_2) {
        std::cout << "shape of A: (" << this->get_shape()[0] << ", " << this->get_shape()[1] << ")" << std::endl;
        std::cout << "shape of B: (" << other.get_shape()[0] << ", " << other.get_shape()[1] << ")" << std::endl;
        throw error::InvalidArgumentError("The inner dimensions of two tensors must be the same.");
    }

    int k = k_1;
    if (this->in_cpu() && other.in_cpu()) {
        Tensor<Tp> out({m, n}, DeviceType::CPU);
        ops::matmul_op<Tp, device::CPU>()(
            device::cpu_device,
            "N", "N",
            m, n, k,
            1.0,
            this->get_data(), k,
            other.get_data(), n,
            0.0,
            out.get_data(), n
        );
        return out;
    } else if (this->in_gpu() && other.in_gpu()) {
        Tensor<Tp> out({m, n}, DeviceType::GPU);
        ops::matmul_op<Tp, device::GPU>()(
            device::gpu_device,
            "N", "N",
            m, n, k,
            1.0,
            this->get_data(), k,
            other.get_data(), n,
            0.0,
            out.get_data(), n
        );
        return out;
    } else {
        throw error::DeviceError("The device of two tensors must be the same.");
    }
}

template <typename Tp>
bool Tensor<Tp>::operator==(const Tensor<Tp>& other) const {
    if (this == &other) {
        return true;
    }
    if (this->get_shape() != other.get_shape()) {
        throw error::InvalidArgumentError("The shape of two tensors must be the same.");
    }

    if (this->in_cpu() && other.in_cpu()) {
        bool out = true;
        ops::equal_op<Tp, device::CPU>()(device::cpu_device, &out, this->get_data(), other.get_data(), this->get_tol_size());
        return out;
    } else if (this->in_gpu() && other.in_gpu()) {
        bool out = true;
        bool* out_device = nullptr;
        size_t tol_size = this->get_tol_size();
        
        Tensor<bool> device_result({1}, DeviceType::GPU);
        out_device = device_result.get_data();
        
        ops::equal_op<Tp, device::GPU>()(device::gpu_device, out_device, this->get_data(), other.get_data(), tol_size);
        memory::copy_mem_op<bool, device::CPU, device::GPU>()(device::cpu_device, device::gpu_device, &out, out_device, 1);

        return out;
    } else {
        throw error::DeviceError("The device of two tensors must be the same.");
    }
}

template <typename Tp>
Tp& Tensor<Tp>::operator[](const std::vector<int>& indices) const {
    int index = get_index(_shape, indices);
    return p_data[index];
}

template <typename Tp>
Tp& Tensor<Tp>::operator[](const std::initializer_list<int>& indices) const {
    return this->operator[](std::vector<int>(indices));
}

template <typename Tp>
int Tensor<Tp>::get_index(const std::vector<int>& shape, const std::vector<int>& indices) {
    if (shape.size() != indices.size()) {
        throw error::InvalidArgumentError("The size of the shape and indices must be the same.");
    }
    int index = 0;
    int dim = 1;
    // the tensor is stored in row-major order
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw error::InvalidArgumentError("The indices must be in the range of the shape.");
        }
        index += indices[i] * dim;
        dim *= shape[i];
    }
    return index;
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::ones(const std::vector<int>& shape, DeviceType device) {
    tensor::Tensor<Tp> out(shape, device);
    int tol_size = out.get_tol_size();
    if (device == DeviceType::CPU) {
        ops::ones_op<Tp, device::CPU>()(device::cpu_device, out.get_data(), tol_size);
        return out;
    } else if (device == DeviceType::GPU) {
        ops::ones_op<Tp, device::GPU>()(device::gpu_device, out.get_data(), tol_size);
        return out;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::ones(const std::initializer_list<int>& shape, DeviceType device) {
    return ones(std::vector<int>(shape), device);
}


template <typename Tp>
void print_tensor_data(
    std::ostream& os, 
    const Tp* data, 
    const std::vector<int>& shape, 
    const std::vector<int>& strides, 
    int dim, 
    int current_dim = 0, 
    int offset = 0
) {
    if (current_dim == dim - 1) {
        os << "[";
        for (int i = 0; i < shape[current_dim]; ++i) {
            os << data[offset + i * strides[current_dim]];
            if (i != shape[current_dim] - 1) os << ", ";
        }
        os << "]";
    } else {
        os << "[";
        for (int i = 0; i < shape[current_dim]; ++i) {
            print_tensor_data(os, data, shape, strides, dim, current_dim + 1, 
                            offset + i * strides[current_dim]);
            if (i != shape[current_dim] - 1) os << ",\n" << std::string(current_dim + 1, ' ');
        }
        os << "]";
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor<double>& tensor) {
    os << "tensor(" << std::endl;
    os << "shape = [";
    for (size_t i = 0; i < tensor.dim(); ++i) {
        os << tensor.get_shape()[i];
        if (i != tensor.dim() - 1) {
            os << ", ";
        }
    }
    os << "], device = ";
    os << tensor.in_cpu() ? "CPU" : "GPU";
    os << ", " << std::endl << "data = " << std::endl;
    
    std::vector<int> strides(tensor.dim());
    int stride = 1;
    for (int i = tensor.dim() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= tensor.get_shape()[i];
    }
    
    print_tensor_data(os, tensor.get_data(), tensor.get_shape(), strides, tensor.dim());
    
    os << ")";
    return os;
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

}