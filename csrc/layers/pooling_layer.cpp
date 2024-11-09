/**
 * @file pooling_layer.cpp
 * @brief Pooling Layer implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "layers/layers.h"
#include "core/kernels/ops.h"

namespace layers {

template <typename Tp>
void max_pool_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, channels, height, width)
    tensor::Tensor<int>& mask,          // mask(batch_size, channels, height_out, width_out)
    tensor::Tensor<Tp>& output,         // Y(batch_size, channels, height_out, width_out)
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
) {
    if (!(input.in_cpu() && mask.in_cpu() && output.in_cpu()) &&
        !(input.in_gpu() && mask.in_gpu() && output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.dim() != 4) {
        throw error::InvalidArgumentError("Input must be a 4D tensor");
    }
    if (mask.dim() != 4) {
        throw error::InvalidArgumentError("Mask must be a 4D tensor");
    }
    if (output.dim() != 4) {
        throw error::InvalidArgumentError("Output must be a 4D tensor");
    }
    if (input.get_shape()[0] != output.get_shape()[0] || input.get_shape()[1] != output.get_shape()[1]) {
        throw error::InvalidArgumentError("Batch size or channels do not match");
    }
    const int batch_size = input.get_shape()[0];
    const int channels = input.get_shape()[1];
    const int height = input.get_shape()[2];
    const int width = input.get_shape()[3];
    const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    if (mask.get_shape()[2] != height_out || mask.get_shape()[3] != width_out) {
        throw error::InvalidArgumentError("Mask shape does not match");
    }
    if (output.get_shape()[2] != height_out || output.get_shape()[3] != width_out) {
        throw error::InvalidArgumentError("Output shape does not match");
    }

    if (input.in_cpu()) {
        ops::max_pool_forward_op<Tp, device::CPU>()(
            device::cpu_device, 
            output.get_data(), 
            mask.get_data(), 
            input.get_data(), 
            batch_size, 
            channels, 
            height, 
            width, 
            kernel_h, 
            kernel_w, 
            pad_h, 
            pad_w, 
            stride_h, 
            stride_w
        );
    } else if (input.in_gpu()) {
        ops::max_pool_forward_op<Tp, device::GPU>()(
            device::gpu_device, 
            output.get_data(), 
            mask.get_data(), 
            input.get_data(), 
            batch_size, 
            channels, 
            height, 
            width, 
            kernel_h, 
            kernel_w, 
            pad_h, 
            pad_w, 
            stride_h, 
            stride_w
        );
    }
}

template <typename Tp>
void max_pool_backward(
    tensor::Tensor<Tp>& grad_input,         // dX(batch_size, channels, height, width)
    const tensor::Tensor<int>& mask,        // mask(batch_size, channels, height_out, width_out)
    const tensor::Tensor<Tp>& grad_output,  // dY(batch_size, channels, height_out, width_out)
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
) {
    if (!(grad_input.in_cpu() && mask.in_cpu() && grad_output.in_cpu()) &&
        !(grad_input.in_gpu() && mask.in_gpu() && grad_output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (grad_input.dim() != 4 || grad_output.dim() != 4) {
        throw error::InvalidArgumentError("Grad input and grad output must be 4D tensors");
    }
    if (
        !(grad_input.get_shape()[0] == grad_output.get_shape()[0] && grad_input.get_shape()[0] == mask.get_shape()[0]) ||
        !(grad_input.get_shape()[1] == grad_output.get_shape()[1] && grad_input.get_shape()[1] == mask.get_shape()[1])
    ) {
        throw error::InvalidArgumentError("Batch size or channels do not match");
    }
    const int batch_size = grad_input.get_shape()[0];
    const int channels = grad_input.get_shape()[1];
    const int height = grad_input.get_shape()[2];
    const int width = grad_input.get_shape()[3];
    const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    if (mask.get_shape()[2] != height_out || mask.get_shape()[3] != width_out) {
        throw error::InvalidArgumentError("Mask shape does not match");
    }
    if (grad_output.get_shape()[2] != height_out || grad_output.get_shape()[3] != width_out) {
        throw error::InvalidArgumentError("Grad output shape does not match");
    }

    if (grad_input.in_cpu()) {
        ops::max_pool_backward_op<Tp, device::CPU>()(
            device::cpu_device, 
            grad_input.get_data(), 
            mask.get_data(), 
            grad_output.get_data(), 
            batch_size, 
            channels, 
            height, 
            width, 
            kernel_h, 
            kernel_w, 
            pad_h, 
            pad_w, 
            stride_h, 
            stride_w
        );
    } else if (grad_input.in_gpu()) {
        ops::max_pool_backward_op<Tp, device::GPU>()(
            device::gpu_device, 
            grad_input.get_data(), 
            mask.get_data(), 
            grad_output.get_data(), 
            batch_size, 
            channels, 
            height, 
            width, 
            kernel_h, 
            kernel_w, 
            pad_h, 
            pad_w, 
            stride_h, 
            stride_w
        );
    }
}

template void max_pool_forward<float>(
    const tensor::Tensor<float>& input,
    tensor::Tensor<int>& mask,
    tensor::Tensor<float>& output,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void max_pool_forward<double>(
    const tensor::Tensor<double>& input,
    tensor::Tensor<int>& mask,
    tensor::Tensor<double>& output,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template void max_pool_backward<float>(
    tensor::Tensor<float>& grad_input,
    const tensor::Tensor<int>& mask,
    const tensor::Tensor<float>& grad_output,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void max_pool_backward<double>(
    tensor::Tensor<double>& grad_input,
    const tensor::Tensor<int>& mask,
    const tensor::Tensor<double>& grad_output,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);


} // namespace layers