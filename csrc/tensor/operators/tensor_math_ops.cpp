/**
 * @file tensor_math_ops.cpp
 * @brief Tensor math operator implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "tensor/tensor.h"
#include "core/memory/memory.h"

#include "error/error.h"

namespace tensor {

template <typename Tp>
Tensor<Tp>& operator+(const Tensor<Tp>& a, const Tensor<Tp>& b) {
    if (a.get_shape() != b.get_shape()) {
        throw error::InvalidArgument("The shape of two tensors must be the same.");
    }

    if (a.in_cpu() && b.in_cpu()) {
        Tensor<Tp> out(a.get_shape(), a.get_device());
        double* p_out;
        size_t tol_size = a.get_tol_size();
        memory::malloc_mem_op<double, device::CPU>()(device::cpu_device, p_out, tol_size);
        ops::add_op<Tp, device::CPU>()(a.device, p_out, a.get_data(), b.get_data(), tol_size);
        out.set_data(p_out, device::cpu_device);
        memory::free_mem_op<double, device::CPU>()(device::cpu_device, p_out);
        return out;
    } else if (a.in_gpu() && b.in_gpu()) {
        Tensor<Tp> out(a.get_shape(), a.get_device());
        double* p_out;
        size_t tol_size = a.get_tol_size();
        memory::malloc_mem_op<double, device::GPU>()(device::gpu_device, p_out, tol_size);
        ops::add_op<Tp, device::GPU>()(a.device, p_out, a.get_data(), b.get_data(), tol_size);
        out.set_data(p_out, device::gpu_device);
        memory::free_mem_op<double, device::GPU>()(device::gpu_device, p_out);
        return out;
    } else {
        throw error::InvalidArgument("The device of two tensors must be the same.");
    }
}

template <typename Tp>
Tensor<Tp>& operator-(const Tensor<Tp>& a, const Tensor<Tp>& b) {
    if (a.get_shape() != b.get_shape()) {
        throw error::InvalidArgument("The shape of two tensors must be the same.");
    }

    if (a.in_cpu() && b.in_cpu()) {
        Tensor<Tp> out(a.get_shape(), a.get_device());
        double* p_out;
        size_t tol_size = a.get_tol_size();
        memory::malloc_mem_op<double, device::CPU>()(device::cpu_device, p_out, tol_size);
        ops::sub_op<Tp, device::CPU>()(a.device, p_out, a.get_data(), b.get_data(), tol_size);
        out.set_data(p_out, device::cpu_device);
        memory::free_mem_op<double, device::CPU>()(device::cpu_device, p_out);
        return out;
    } else if (a.in_gpu() && b.in_gpu()) {
        Tensor<Tp> out(a.get_shape(), a.device);
        double* p_out;
        size_t tol_size = a.get_tol_size();
        memory::malloc_mem_op<double, device::GPU>()(device::gpu_device, p_out, tol_size);
        ops::sub_op<Tp, device::GPU>()(a.device, p_out, a.get_data(), b.get_data(), tol_size);
        out.set_data(p_out, device::gpu_device);
        memory::free_mem_op<double, device::GPU>()(device::gpu_device, p_out);
        return out;
    } else {
        throw error::InvalidArgument("The device of two tensors must be the same.");
    }
}

template <typename Tp>
bool operator==(const Tensor<Tp>& a, const Tensor<Tp>& b) {
    if (a.get_shape() != b.get_shape()) {
        throw error::InvalidArgument("The shape of two tensors must be the same.");
    }

    if (a.in_cpu() && b.in_cpu()) {
        bool out;
        ops::equal_op<Tp, device::CPU>()(a.device, &out, a.get_data(), b.get_data(), a.get_tol_size());
        return out;
    } else if (a.in_gpu() && b.in_gpu()) {
        bool out_c;
        bool* out_g;
        size_t tol_size = a.get_tol_size();
        memory::malloc_mem_op<bool, device::GPU>()(device::gpu_device, out_g, 1);
        ops::equal_op<Tp, device::GPU>()(a.device, out_g, a.get_data(), b.get_data(), tol_size);
        memory::copy_mem_op<bool, device::CPU, device::GPU>()(device::cpu_device, device::gpu_device, &out_c, out_g, 1);
        memory::free_mem_op<bool, device::GPU>()(device::gpu_device, out_g);
        return out_c;
    } else {
        throw error::InvalidArgument("The device of two tensors must be the same.");
    }
}

} // namespace tensor