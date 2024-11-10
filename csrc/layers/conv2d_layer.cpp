/**
 * @file conv2d_layer.cpp
 * @brief Conv2d Layer implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "error/error.h"
#include "layers/layers.h"
#include "core/kernels/ops.h"

namespace layers {

template <typename Tp>
void conv2d_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, in_channels, height, width)
    const tensor::Tensor<Tp>& weight,   // W(out_channels, in_channels, kernel_h, kernel_w)
    const tensor::Tensor<Tp>& bias,     // b(out_channels)
    tensor::Tensor<Tp>& output,         // Y(batch_size, out_channels, height_out, width_out)
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
) {
    if (
        !(input.in_cpu() && weight.in_cpu() && bias.in_cpu() && output.in_cpu()) &&
        !(input.in_gpu() && weight.in_gpu() && bias.in_gpu() && output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.dim() != 4) {
        throw error::InvalidArgumentError("Input must be a 4D tensor");
    }
    const int batch_size = input.get_shape()[0];
    const int in_channels = input.get_shape()[1];
    const int height = input.get_shape()[2];
    const int width = input.get_shape()[3];
    const int out_channels = weight.get_shape()[0];
    const int kernel_h = weight.get_shape()[2];
    const int kernel_w = weight.get_shape()[3];
    const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    if (weight.dim() != 4 || weight.get_shape()[1] != in_channels) {
        throw error::InvalidArgumentError("Weight must be a 4D tensor with in_channels in the second dimension");
    }
    if (bias.dim() != 1 || bias.get_shape()[0] != out_channels) {
        throw error::InvalidArgumentError("Bias must be a 1D tensor with out_channels in the first dimension");
    }
    if (output.dim() != 4 || output.get_shape()[0] != batch_size || output.get_shape()[1] != out_channels || output.get_shape()[2] != height_out || output.get_shape()[3] != width_out) {
        throw error::InvalidArgumentError("Output must be a 4D tensor with batch_size in the first dimension, out_channels in the second dimension, height_out in the third dimension, and width_out in the fourth dimension");
    }

    if (input.in_cpu()) {
        ops::conv2d_forward_op<Tp, device::CPU>()(
            device::cpu_device,
            output.get_data(),
            input.get_data(),
            weight.get_data(),
            bias.get_data(),
            batch_size, in_channels, out_channels, 
            height, width, 
            kernel_h, kernel_w, 
            pad_h, pad_w, 
            stride_h, stride_w
        );
    } else if (input.in_gpu()) {
        ops::conv2d_forward_op<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            weight.get_data(),
            bias.get_data(),
            batch_size, in_channels, out_channels, 
            height, width, 
            kernel_h, kernel_w, 
            pad_h, pad_w, 
            stride_h, stride_w
        );
    }
}

template <typename Tp>
void conv2d_backward(
    const tensor::Tensor<Tp>& input,        // X(batch_size, in_channels, height, width)
    const tensor::Tensor<Tp>& weight,       // W(out_channels, in_channels, kernel_h, kernel_w)
    tensor::Tensor<Tp>& grad_input,         // dX(batch_size, in_channels, height, width)
    tensor::Tensor<Tp>& grad_weight,        // dW(out_channels, in_channels, kernel_h, kernel_w)
    tensor::Tensor<Tp>& grad_bias,          // db(out_channels)
    const tensor::Tensor<Tp>& grad_output,  // dY(batch_size, out_channels, height_out, width_out)
    const int pad_h,    
    const int pad_w,
    const int stride_h,
    const int stride_w
) {
    if (
        !(input.in_cpu() && weight.in_cpu() && grad_input.in_cpu() && grad_weight.in_cpu() && grad_bias.in_cpu() && grad_output.in_cpu()) &&
        !(input.in_gpu() && weight.in_gpu() && grad_input.in_gpu() && grad_weight.in_gpu() && grad_bias.in_gpu() && grad_output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.dim() != 4 || grad_output.dim() != 4 || weight.dim() != 4) {
        throw error::InvalidArgumentError("Input, grad_output and weight must be a 4D tensor");
    }
    if (grad_input.dim() != 4 || grad_weight.dim() != 4 || grad_bias.dim() != 1) {
        throw error::InvalidArgumentError("grad_input, grad_weight and grad_bias must be a 4D tensor and a 1D tensor respectively");
    }
    const int batch_size = input.get_shape()[0];
    const int in_channels = input.get_shape()[1];
    const int height = input.get_shape()[2];
    const int width = input.get_shape()[3];
    const int out_channels = weight.get_shape()[0];
    const int kernel_h = weight.get_shape()[2];
    const int kernel_w = weight.get_shape()[3];
    const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    if (grad_input.get_shape()[0] != batch_size || grad_input.get_shape()[1] != in_channels || grad_input.get_shape()[2] != height || grad_input.get_shape()[3] != width) {
        throw error::InvalidArgumentError("grad_input must be a 4D tensor with batch_size in the first dimension, in_channels in the second dimension, height in the third dimension, and width in the fourth dimension");
    }
    if (grad_weight.get_shape()[0] != out_channels || grad_weight.get_shape()[1] != in_channels || grad_weight.get_shape()[2] != kernel_h || grad_weight.get_shape()[3] != kernel_w) {
        throw error::InvalidArgumentError("grad_weight must be a 4D tensor with out_channels in the first dimension, in_channels in the second dimension, kernel_h in the third dimension, and kernel_w in the fourth dimension");
    }
    if (grad_bias.get_shape()[0] != out_channels) {
        throw error::InvalidArgumentError("grad_bias must be a 1D tensor with out_channels in the first dimension");
    }
    if (grad_output.get_shape()[0] != batch_size || grad_output.get_shape()[1] != out_channels || grad_output.get_shape()[2] != height_out || grad_output.get_shape()[3] != width_out) {
        throw error::InvalidArgumentError("grad_output must be a 4D tensor with batch_size in the first dimension, out_channels in the second dimension, height_out in the third dimension, and width_out in the fourth dimension");
    }

    if (input.in_cpu()) {
        ops::conv2d_backward_op<Tp, device::CPU>()(
            device::cpu_device,
            grad_input.get_data(),
            grad_weight.get_data(),
            grad_bias.get_data(),
            grad_output.get_data(),
            input.get_data(),
            weight.get_data(),
            batch_size, in_channels, out_channels, 
            height, width, 
            kernel_h, kernel_w, 
            pad_h, pad_w, 
            stride_h, stride_w
        );
    } else if (input.in_gpu()) {
        ops::conv2d_backward_op<Tp, device::GPU>()(
            device::gpu_device,
            grad_input.get_data(),
            grad_weight.get_data(),
            grad_bias.get_data(),
            grad_output.get_data(),
            input.get_data(),
            weight.get_data(),
            batch_size, in_channels, out_channels, 
            height, width, 
            kernel_h, kernel_w, 
            pad_h, pad_w, 
            stride_h, stride_w
        );
    }
}

template void conv2d_forward<float>(
    const tensor::Tensor<float>& input,
    const tensor::Tensor<float>& weight,
    const tensor::Tensor<float>& bias,
    tensor::Tensor<float>& output,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w
);
template void conv2d_forward<double>(
    const tensor::Tensor<double>& input,
    const tensor::Tensor<double>& weight,
    const tensor::Tensor<double>& bias,
    tensor::Tensor<double>& output,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w
);

template void conv2d_backward<float>(
    const tensor::Tensor<float>& input,
    const tensor::Tensor<float>& weight,
    tensor::Tensor<float>& grad_input,
    tensor::Tensor<float>& grad_weight,
    tensor::Tensor<float>& grad_bias,
    const tensor::Tensor<float>& grad_output,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w
);
template void conv2d_backward<double>(
    const tensor::Tensor<double>& input,
    const tensor::Tensor<double>& weight,
    tensor::Tensor<double>& grad_input,
    tensor::Tensor<double>& grad_weight,
    tensor::Tensor<double>& grad_bias,
    const tensor::Tensor<double>& grad_output,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w
);

} // namespace layers