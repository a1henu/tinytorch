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

} // namespace layers