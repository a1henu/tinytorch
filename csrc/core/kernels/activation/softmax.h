/**
 * @file softmax.h
 * @brief softmax operator declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_ACTIVATION_SOFTMAX_H
#define CSRC_CORE_KERNELS_ACTIVATION_SOFTMAX_H

#include <cstddef>

#include "core/device/device.h"
#include "macros.h"

namespace ops {

template <typename Tp, typename Device>
struct softmax_forward {
    /// @brief softmax forward operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input  : the input array pointer
    /// @param size   : the size of the array 
    void operator()(
        Device* device, 
        Tp* output, 
        const Tp* input, 
        size_t batch_size, 
        size_t num_classes
    );
};

} // namespace ops

#endif