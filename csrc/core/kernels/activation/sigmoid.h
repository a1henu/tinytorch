/**
 * @file sigmoid.h
 * @brief sigmoid operator declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_ACTIVATION_SIGMOID_H
#define CSRC_CORE_KERNELS_ACTIVATION_SIGMOID_H

#include <cstddef>

#include "core/device/device.h"

namespace activation {

template <typename Tp, typename Device>
struct sigmoid_forward {
    /// @brief sigmoid forward operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input  : the input array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, Tp* input, size_t size);
};

template <typename Tp, typename Device>
struct sigmoid_backward {
    /// @brief sigmoid backward operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input  : the input array pointer
    /// @param grad   : the gradient array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, Tp* input, Tp* grad, size_t size);
};

} // namespace activation

#endif