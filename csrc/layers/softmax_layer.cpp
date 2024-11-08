/**
 * @file softmax_layers.cpp
 * @brief Softmax Layers implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "error/error.h"
#include "layers/layers.h"
#include "core/kernels/functions/softmax.h"

namespace layers {

template <typename Tp>
void softmax_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    tensor::Tensor<Tp>& output          // Y(batch_size, num_classes)
) {
    if(
        !(input.in_cpu() && output.in_cpu()) &&
        !(input.in_gpu() && output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }

    if (input.get_shape().size() != 2) {
        throw error::InvalidArgumentError("Input must be a 2D tensor");
    }

    if (output.get_shape().size() != 2) {
        throw error::InvalidArgumentError("Output must be a 2D tensor");
    }

    if (input.get_shape()[0] != output.get_shape()[0] || input.get_shape()[1] != output.get_shape()[1]) {
        throw error::InvalidArgumentError("Input and output dimensions do not match");
    }
    size_t batch_size = input.get_shape()[0];
    size_t num_classes = input.get_shape()[1];

    if (input.in_cpu()) {
        ops::softmax_forward<Tp, device::CPU>()(
            device::cpu_device,
            output.get_data(),
            input.get_data(),
            batch_size,
            num_classes
        );
    } else if (input.in_gpu()) {
        ops::softmax_forward<Tp, device::GPU>()(
            device::gpu_device,
            output.get_data(),
            input.get_data(),
            batch_size,
            num_classes
        );
    } else {
        throw error::InvalidArgumentError("Invalid device type");
    }
}

template void softmax_forward<float>(
    const tensor::Tensor<float>& input,    // X(batch_size, num_classes)
    tensor::Tensor<float>& output          // Y(batch_size, num_classes)
);
template void softmax_forward<double>(
    const tensor::Tensor<double>& input,    // X(batch_size, num_classes)
    tensor::Tensor<double>& output          // Y(batch_size, num_classes)
);

} // namespace layers