/**
 * @file mse.h
 * @brief Mean Squared Error operator declaration
 * 
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_FUNCTIONS_MSE_H
#define CSRC_CORE_KERNELS_FUNCTIONS_MSE_H

#include <cstddef>

#include "core/device/device.h"
#include "macros.h"

namespace ops {

template <typename Tp, typename Device>
struct mse_forward {
    /// @brief mse forward operator for multi-device
    ///
    /// Inputs:
    /// @param device      : the type of device
    /// @param output      : the output array pointer
    /// @param input       : the input array pointer
    /// @param target      : the target array pointer
    /// @param batch_size  : the size of the batch
    /// @param num_classes : the number of classes
    void operator()(
        Device* device, 
        Tp* output, 
        const Tp* input, 
        const Tp* target, 
        size_t batch_size,
        size_t num_classes
    );
};

template <typename Tp, typename Device>
struct mse_backward {
    /// @brief mse backward operator for multi-device
    ///
    /// Inputs:
    /// @param device      : the type of device
    /// @param output      : the output array pointer
    /// @param input       : the input array pointer
    /// @param target      : the target array pointer
    /// @param batch_size  : the size of the batch
    /// @param num_classes : the number of classes
    void operator()(
        Device* device, 
        Tp* output, 
        const Tp* input, 
        const Tp* target, 
        size_t batch_size,
        size_t num_classes
    );
};

} // namespace ops

#endif