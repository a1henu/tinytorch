/**
 * @file cross_entropy.h
 * @brief cross entropy operator declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_FUNCTIONS_CROSS_ENTROPY_H
#define CSRC_CORE_KERNELS_FUNCTIONS_CROSS_ENTROPY_H

#include <cstddef>

#include "core/device/device.h"
#include "macros.h"

namespace ops {

template <typename Tp, typename Device>
struct cross_entropy_forward {
    /// @brief cross entropy forward operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output scalar pointer
    /// @param input  : the input array pointer  (N, C)
    /// @param target : the target array pointer (N)
    /// @param batch_size   : the number of samples
    /// @param num_classes  : the number of classes
    void operator()(
        Device* device, 
        Tp* output, 
        const Tp* input, 
        const int* target, 
        size_t batch_size,
        size_t num_classes
    );
};

} // namespace ops

#endif