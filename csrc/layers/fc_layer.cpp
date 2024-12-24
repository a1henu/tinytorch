/**
 * @file fc_layers.cpp
 * @brief Fully Connected Layers implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "error/error.h"
#include "layers/layers.h"
#include "core/kernels/ops.h"

namespace layers {

template <typename Tp>
void fc_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, in_features)
    const tensor::Tensor<Tp>& weight,   // W(in_features, out_features)
    const tensor::Tensor<Tp>& bias,     // b(1, out_features)
    tensor::Tensor<Tp>& output          // Y(batch_size, out_features)
) {
    if(
        !(input.in_cpu() && weight.in_cpu() && bias.in_cpu() && output.in_cpu()) &&
        !(input.in_gpu() && weight.in_gpu() && bias.in_gpu() && output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }

    if (input.get_shape()[1] != weight.get_shape()[0]) {
        throw error::InvalidArgumentError("Input and weight dimensions do not match");
    }

    if (weight.get_shape()[1] != bias.get_shape()[1]) {
        throw error::InvalidArgumentError("Weight and bias dimensions do not match");
    }

    if (input.get_shape()[0] != output.get_shape()[0]) {
        throw error::InvalidArgumentError("Batch size does not match");
    }

    if (weight.get_shape()[1] != output.get_shape()[1]) {
        throw error::InvalidArgumentError("Output dimensions do not match");
    }

    if (input.in_cpu()) {
        ops::fc_forward_op<Tp, device::CPU>()(
            device::cpu_device,
            output.get_data(),
            input.get_data(),
            weight.get_data(),
            bias.get_data(),
            input.get_shape()[0],
            input.get_shape()[1],
            weight.get_shape()[1]
        );
    } else if (input.in_gpu()) {
        ops::fc_forward_op<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            weight.get_data(),
            bias.get_data(),
            input.get_shape()[0],
            input.get_shape()[1],
            weight.get_shape()[1]
        );
    }
}

template <typename Tp>
void fc_backward(
    const tensor::Tensor<Tp>& input,        // X(batch_size, in_features)
    const tensor::Tensor<Tp>& weight,       // W(in_features, out_features)
    const tensor::Tensor<Tp>& bias,         // b(1, out_features)
    const tensor::Tensor<Tp>& output,       // Y(batch_size, out_features)
    tensor::Tensor<Tp>& grad_input,         // dX(batch_size, in_features)
    tensor::Tensor<Tp>& grad_weight,        // dW(in_features, out_features)
    tensor::Tensor<Tp>& grad_bias,          // db(1, out_features)
    const tensor::Tensor<Tp>& grad_output   // dY(batch_size, out_features)
) {
    if(
        !(input.in_cpu() && weight.in_cpu() && bias.in_cpu() && output.in_cpu() && grad_input.in_cpu() && grad_weight.in_cpu() && grad_bias.in_cpu() && grad_output.in_cpu()) &&
        !(input.in_gpu() && weight.in_gpu() && bias.in_gpu() && output.in_gpu() && grad_input.in_gpu() && grad_weight.in_gpu() && grad_bias.in_gpu() && grad_output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }

    if (input.get_shape()[1] != weight.get_shape()[0]) {
        throw error::InvalidArgumentError("Input and weight dimensions do not match");
    }

    if (weight.get_shape()[1] != bias.get_shape()[1]) {
        throw error::InvalidArgumentError("Weight and bias dimensions do not match");
    }

    if (input.get_shape()[0] != output.get_shape()[0]) {
        throw error::InvalidArgumentError("Batch size does not match");
    }

    if (weight.get_shape()[1] != output.get_shape()[1]) {
        throw error::InvalidArgumentError("Output dimensions do not match");
    }

    if (input.in_cpu()) {
        ops::fc_backward_op<Tp, device::CPU>()(
            device::cpu_device,
            grad_input.get_data(),
            grad_weight.get_data(),
            grad_bias.get_data(),
            grad_output.get_data(),
            input.get_data(),
            weight.get_data(),
            input.get_shape()[0],
            input.get_shape()[1],
            weight.get_shape()[1]
        );
    } else if (input.in_gpu()) {
        ops::fc_backward_op<Tp, device::GPU>()(
            device::gpu_device,
            grad_input.get_data(),
            grad_weight.get_data(),
            grad_bias.get_data(),
            grad_output.get_data(),
            input.get_data(),
            weight.get_data(),
            input.get_shape()[0],
            input.get_shape()[1],
            weight.get_shape()[1]
        );
    }
}

template void fc_forward<float>(
    const tensor::Tensor<float>& input,    // X(batch_size, in_features)
    const tensor::Tensor<float>& weight,   // W(in_features, out_features)
    const tensor::Tensor<float>& bias,     // b(1, out_features)
    tensor::Tensor<float>& output          // Y(batch_size, out_features)
);
template void fc_forward<double>(
    const tensor::Tensor<double>& input,    // X(batch_size, in_features)
    const tensor::Tensor<double>& weight,   // W(in_features, out_features)
    const tensor::Tensor<double>& bias,     // b(1, out_features)
    tensor::Tensor<double>& output          // Y(batch_size, out_features)
);

template void fc_backward<float>(
    const tensor::Tensor<float>& input,        // X(batch_size, in_features)
    const tensor::Tensor<float>& weight,       // W(in_features, out_features)
    const tensor::Tensor<float>& bias,         // b(1, out_features)
    const tensor::Tensor<float>& output,       // Y(batch_size, out_features)
    tensor::Tensor<float>& grad_input,         // dX(batch_size, in_features)
    tensor::Tensor<float>& grad_weight,        // dW(in_features, out_features)
    tensor::Tensor<float>& grad_bias,          // db(1, out_features)
    const tensor::Tensor<float>& grad_output   // dY(batch_size, out_features)
);
template void fc_backward<double>(
    const tensor::Tensor<double>& input,        // X(batch_size, in_features)
    const tensor::Tensor<double>& weight,       // W(in_features, out_features)
    const tensor::Tensor<double>& bias,         // b(1, out_features)
    const tensor::Tensor<double>& output,       // Y(batch_size, out_features)
    tensor::Tensor<double>& grad_input,         // dX(batch_size, in_features)
    tensor::Tensor<double>& grad_weight,        // dW(in_features, out_features)
    tensor::Tensor<double>& grad_bias,          // db(1, out_features)
    const tensor::Tensor<double>& grad_output   // dY(batch_size, out_features)
);

} // namespace layers
