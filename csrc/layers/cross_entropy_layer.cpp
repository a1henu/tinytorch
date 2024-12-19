/**
 * @file cross_entropy_layers.cpp
 * @brief Cross Entropy Layers implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "error/error.h"
#include "layers/layers.h"
#include "core/kernels/functions/cross_entropy.h"

namespace layers {

template <typename Tp>
void cross_entropy_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<Tp>& target,  // t(batch_size)
    tensor::Tensor<Tp>& output          // z(1)
) {
    if (!(input.in_cpu() && target.in_cpu() && output.in_cpu()) &&
        !(input.in_gpu() && target.in_gpu() && output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.get_shape()[0] != target.get_shape()[0]) {
        throw error::InvalidArgumentError("Batch size does not match");
    }
    if (target.dim() != 1) {
        throw error::InvalidArgumentError("Target must be a 1D tensor");
    }
    if (!(output.dim() == 1 && output.get_shape()[0] == 1)) {
        throw error::InvalidArgumentError("Output must be a 1D scalar");
    }

    if (input.in_cpu()) {
        ops::cross_entropy_forward<Tp, device::CPU>()(
            device::cpu_device, 
            output.get_data(), 
            input.get_data(), 
            reinterpret_cast<const Tp*>(target.get_data()), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    } else if (input.in_gpu()) {
        ops::cross_entropy_forward<Tp, device::GPU>()(
            device::gpu_device, 
            output.get_data(), 
            input.get_data(), 
            reinterpret_cast<const Tp*>(target.get_data()), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    }
}

template <typename Tp>
void cross_entropy_backward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<Tp>& target,   // t(batch_size)
    tensor::Tensor<Tp>& grad            // dX(batch_size, num_classes)
) {
    if (!(input.in_cpu() && target.in_cpu() && grad.in_cpu()) &&
        !(input.in_gpu() && target.in_gpu() && grad.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.get_shape()[0] != target.get_shape()[0]) {
        throw error::InvalidArgumentError("Batch size does not match");
    }
    if (target.dim() != 1) {
        throw error::InvalidArgumentError("Target must be a 1D tensor");
    }
    if (grad.get_shape()[0] != input.get_shape()[0] || grad.get_shape()[1] != input.get_shape()[1]) {
        throw error::InvalidArgumentError("Gradient dimensions do not match");
    }

    if (input.in_cpu()) {
        ops::cross_entropy_backward<Tp, device::CPU>()(
            device::cpu_device, 
            grad.get_data(), 
            input.get_data(), 
            target.get_data(), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    } else if (input.in_gpu()) {
        ops::cross_entropy_backward<Tp, device::GPU>()(
            device::gpu_device, 
            grad.get_data(), 
            input.get_data(), 
           target.get_data(), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    }
}

template void cross_entropy_forward<float>(
    const tensor::Tensor<float>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<float>& target,   // t(batch_size)
    tensor::Tensor<float>& output          // z(1)
);
template void cross_entropy_forward<double>(
    const tensor::Tensor<double>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<double>& target,   // t(batch_size)
    tensor::Tensor<double>& output          // z(1)
);

template void cross_entropy_backward<float>(
    const tensor::Tensor<float>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<float>& target,   // t(batch_size)
    tensor::Tensor<float>& grad            // dX(batch_size, num_classes)
);
template void cross_entropy_backward<double>(
    const tensor::Tensor<double>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<double>& target,   // t(batch_size)
    tensor::Tensor<double>& grad            // dX(batch_size, num_classes)
);


} // namespace layers