/**
 * @file mse_layer.cpp
 * @brief Mean Squared Error Layers implementation
 * 
 * Licensed under the MIT License.
 */

#include "error/error.h"
#include "layers/layers.h"
#include "core/kernels/functions/mse.h"

namespace layers {

template <typename Tp>
void mse_forward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<Tp>& target,   // t(batch_size, num_classes)
    tensor::Tensor<Tp>& output          // z(1)
) {
    if (!(input.in_cpu() && target.in_cpu() && output.in_cpu()) &&
        !(input.in_gpu() && target.in_gpu() && output.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.get_shape() != target.get_shape()) {
        throw error::InvalidArgumentError("Input and target shapes do not match");
    }
    if (!(output.dim() == 1 && output.get_shape()[0] == 1)) {
        throw error::InvalidArgumentError("Output must be a 1D scalar");
    }

    if (input.in_cpu()) {
        ops::mse_forward<Tp, device::CPU>()(
            device::cpu_device, 
            output.get_data(), 
            input.get_data(), 
            target.get_data(), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    } else if (input.in_gpu()) {
        ops::mse_forward<Tp, device::GPU>()(
            device::gpu_device, 
            output.get_data(), 
            input.get_data(), 
            target.get_data(), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    }
}

template <typename Tp>
void mse_backward(
    const tensor::Tensor<Tp>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<Tp>& target,   // t(batch_size, num_classes)
    tensor::Tensor<Tp>& grad            // dX(batch_size, num_classes)
) {
    if (!(input.in_cpu() && target.in_cpu() && grad.in_cpu()) &&
        !(input.in_gpu() && target.in_gpu() && grad.in_gpu())
    ) {
        throw error::InvalidArgumentError("All tensors must be in the same device");
    }
    if (input.get_shape() != target.get_shape()) {
        throw error::InvalidArgumentError("Input and target shapes do not match");
    }
    if (grad.get_shape() != input.get_shape()) {
        throw error::InvalidArgumentError("Gradient dimensions do not match");
    }

    if (input.in_cpu()) {
        ops::mse_backward<Tp, device::CPU>()(
            device::cpu_device, 
            grad.get_data(), 
            input.get_data(), 
            target.get_data(), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    } else if (input.in_gpu()) {
        ops::mse_backward<Tp, device::GPU>()(
            device::gpu_device, 
            grad.get_data(), 
            input.get_data(), 
            target.get_data(), 
            input.get_shape()[0], 
            input.get_shape()[1]
        );
    }
}

template void mse_forward<float>(
    const tensor::Tensor<float>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<float>& target,   // t(batch_size, num_classes)
    tensor::Tensor<float>& output          // z(1)
);
template void mse_forward<double>(
    const tensor::Tensor<double>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<double>& target,   // t(batch_size, num_classes)
    tensor::Tensor<double>& output          // z(1)
);

template void mse_backward<float>(
    const tensor::Tensor<float>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<float>& target,   // t(batch_size, num_classes)
    tensor::Tensor<float>& grad            // dX(batch_size, num_classes)
);
template void mse_backward<double>(
    const tensor::Tensor<double>& input,    // X(batch_size, num_classes)
    const tensor::Tensor<double>& target,   // t(batch_size, num_classes)
    tensor::Tensor<double>& grad            // dX(batch_size, num_classes)
);

} // namespace layers