/**
 * @file tensor_activation.cpp
 * @brief Tensor activation function implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"
#include "core/memory/memory.h"
#include "error/error.h"
#include "tensor/tensor.h"

#include "core/kernels/activation/relu.h"
#include "core/kernels/activation/sigmoid.h"

#include "tensor/operators/tensor_activation.h"

template <typename Tp>
tensor::Tensor<Tp> t_relu_f(const tensor::Tensor<Tp>& input) {
    size_t size = input.get_tol_size();
    if (input.in_cpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::CPU>()(
            device::cpu_device, output, size
        );
        ops::relu_forward<Tp, device::CPU>()(
            device::cpu_device, output, input.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::cpu_device, 
            device::cpu_device
        );
        memory::free_mem_op<Tp, device::CPU>()(device::cpu_device, output);

        return output_t;
    } else if (input.in_gpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::GPU>()(
            device::gpu_device, output, size
        );
        ops::relu_forward<Tp, device::GPU>()(
            device::gpu_device, output, input.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::gpu_device, 
            device::gpu_device
        );
        memory::free_mem_op<Tp, device::GPU>()(device::gpu_device, output);

        return output_t;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
tensor::Tensor<Tp> t_relu_b(const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad) {
    size_t size = input.get_tol_size();
    if (input.in_cpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::CPU>()(
            device::cpu_device, output, size
        );
        ops::relu_backward<Tp, device::CPU>()(
            device::cpu_device, output, input.get_data(), grad.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::cpu_device, 
            device::cpu_device
        );
        memory::free_mem_op<Tp, device::CPU>()(device::cpu_device, output);

        return output_t;
    } else if (input.in_gpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::GPU>()(
            device::gpu_device, output, size
        );
        ops::relu_backward<Tp, device::GPU>()(
            device::gpu_device, output, input.get_data(), grad.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::gpu_device, 
            device::gpu_device
        );
        memory::free_mem_op<Tp, device::GPU>()(device::gpu_device, output);
    
        return output_t;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
tensor::Tensor<Tp> t_sigmoid_f(const tensor::Tensor<Tp>& input) {
    size_t size = input.get_tol_size();
    if (input.in_cpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::CPU>()(
            device::cpu_device, output, size
        );
        ops::sigmoid_forward<Tp, device::CPU>()(
            device::cpu_device, output, input.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::cpu_device, 
            device::cpu_device
        );
        memory::free_mem_op<Tp, device::CPU>()(device::cpu_device, output);

        return output_t;
    } else if (input.in_gpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::GPU>()(
            device::gpu_device, output, size
        );
        ops::sigmoid_forward<Tp, device::GPU>()(
            device::gpu_device, output, input.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::gpu_device, 
            device::gpu_device
        );
        memory::free_mem_op<Tp, device::GPU>()(device::gpu_device, output);

        return output_t;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
tensor::Tensor<Tp> t_sigmoid_b(const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad) {
    size_t size = input.get_tol_size();
    if (input.in_cpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::CPU>()(
            device::cpu_device, output, size
        );
        ops::sigmoid_backward<Tp, device::CPU>()(
            device::cpu_device, output, input.get_data(), grad.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::cpu_device, 
            device::cpu_device
        );
        memory::free_mem_op<Tp, device::CPU>()(device::cpu_device, output);

        return output_t;
    } else if (input.in_gpu()) {
        Tp* output;
        memory::malloc_mem_op<Tp, device::GPU>()(
            device::gpu_device, output, size
        );
        ops::sigmoid_backward<Tp, device::GPU>()(
            device::gpu_device, output, input.get_data(), grad.get_data(), size
        );
        tensor::Tensor<Tp> output_t(
            input.get_shape(), 
            output, 
            device::gpu_device, 
            device::gpu_device
        );
        memory::free_mem_op<Tp, device::GPU>()(device::gpu_device, output);

        return output_t;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}
