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
            new device::CPU(), 
            output
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
            new device::GPU(),
            output
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
            new device::CPU(),
            output
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
            new device::GPU(),
            output
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
            new device::CPU(),
            output
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
            new device::GPU(),
            output
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
            new device::CPU(), 
            output
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
            new device::GPU(),
            output
        );
        memory::free_mem_op<Tp, device::GPU>()(device::gpu_device, output);

        return output_t;
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template tensor::Tensor<int> t_relu_f(const tensor::Tensor<int>& input);
template tensor::Tensor<float> t_relu_f(const tensor::Tensor<float>& input);
template tensor::Tensor<double> t_relu_f(const tensor::Tensor<double>& input);

template tensor::Tensor<int> t_relu_b(const tensor::Tensor<int>& input, const tensor::Tensor<int>& grad);
template tensor::Tensor<float> t_relu_b(const tensor::Tensor<float>& input, const tensor::Tensor<float>& grad);
template tensor::Tensor<double> t_relu_b(const tensor::Tensor<double>& input, const tensor::Tensor<double>& grad);

template tensor::Tensor<int> t_sigmoid_f(const tensor::Tensor<int>& input);
template tensor::Tensor<float> t_sigmoid_f(const tensor::Tensor<float>& input);
template tensor::Tensor<double> t_sigmoid_f(const tensor::Tensor<double>& input);

template tensor::Tensor<int> t_sigmoid_b(const tensor::Tensor<int>& input, const tensor::Tensor<int>& grad);
template tensor::Tensor<float> t_sigmoid_b(const tensor::Tensor<float>& input, const tensor::Tensor<float>& grad);
template tensor::Tensor<double> t_sigmoid_b(const tensor::Tensor<double>& input, const tensor::Tensor<double>& grad);