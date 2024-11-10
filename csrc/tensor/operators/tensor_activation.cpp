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

#include "core/kernels/functions/relu.h"
#include "core/kernels/functions/sigmoid.h"

#include "tensor/operators/tensor_activation.h"

namespace tensor {

template <typename Tp>
void relu_forward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input) {
    if (output.get_shape() != input.get_shape()) {
        throw error::InvalidArgumentError("Output shape must be the same as input shape");
    }
    
    const size_t size = input.get_tol_size();

    if (input.in_cpu()) {
        ops::relu_forward<Tp, device::CPU>()(
            device::cpu_device, 
            output.get_data(), 
            input.get_data(), 
            size
        );
    } else if (input.in_gpu()) {
        ops::relu_forward<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            size
        );
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
void relu_backward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad) {
    if (output.get_shape() != input.get_shape()) {
        throw error::InvalidArgumentError("Output shape must be the same as input shape");
    }

    const size_t size = input.get_tol_size();
    
    if (input.in_cpu()) {
        ops::relu_backward<Tp, device::CPU>()(
            device::cpu_device,
            output.get_data(),
            input.get_data(),
            grad.get_data(),
            size
        );
    } else if (input.in_gpu()) {
        ops::relu_backward<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            grad.get_data(),
            size
        );
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
void sigmoid_forward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input) {
    if (output.get_shape() != input.get_shape()) {
        throw error::InvalidArgumentError("Output shape must be the same as input shape");
    }

    const size_t size = input.get_tol_size();
    
    if (input.in_cpu()) {
        ops::sigmoid_forward<Tp, device::CPU>()(
            device::cpu_device,
            output.get_data(),
            input.get_data(),
            size
        );
    } else if (input.in_gpu()) {
        ops::sigmoid_forward<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            size
        );
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template <typename Tp>
void sigmoid_backward(tensor::Tensor<Tp>& output, const tensor::Tensor<Tp>& input, const tensor::Tensor<Tp>& grad) {
    if (output.get_shape() != input.get_shape()) {
        throw error::InvalidArgumentError("Output shape must be the same as input shape");
    }

    const size_t size = input.get_tol_size();
    
    if (input.in_cpu()) {
        ops::sigmoid_backward<Tp, device::CPU>()(
            device::cpu_device,
            output.get_data(),
            input.get_data(),
            grad.get_data(),
            size
        );
    } else if (input.in_gpu()) {
        ops::sigmoid_backward<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            grad.get_data(),
            size
        );
    } else {
        throw error::DeviceError("Unknown device type");
    }
}

template void relu_forward(tensor::Tensor<int>&, const tensor::Tensor<int>&);
template void relu_forward(tensor::Tensor<float>&, const tensor::Tensor<float>&);
template void relu_forward(tensor::Tensor<double>&, const tensor::Tensor<double>&);

template void relu_backward(tensor::Tensor<int>&, const tensor::Tensor<int>&, const tensor::Tensor<int>&);
template void relu_backward(tensor::Tensor<float>&, const tensor::Tensor<float>&, const tensor::Tensor<float>&);
template void relu_backward(tensor::Tensor<double>&, const tensor::Tensor<double>&, const tensor::Tensor<double>&);

template void sigmoid_forward(tensor::Tensor<int>&, const tensor::Tensor<int>&);
template void sigmoid_forward(tensor::Tensor<float>&, const tensor::Tensor<float>&);
template void sigmoid_forward(tensor::Tensor<double>&, const tensor::Tensor<double>&);

template void sigmoid_backward(tensor::Tensor<int>&, const tensor::Tensor<int>&, const tensor::Tensor<int>&);
template void sigmoid_backward(tensor::Tensor<float>&, const tensor::Tensor<float>&, const tensor::Tensor<float>&);
template void sigmoid_backward(tensor::Tensor<double>&, const tensor::Tensor<double>&, const tensor::Tensor<double>&);

} // namespace tensor