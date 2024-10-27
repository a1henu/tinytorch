/**
 * @file relu.h
 * @brief relu operator declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_ACTIVATION_RELU_H
#define CSRC_CORE_KERNELS_ACTIVATION_RELU_H

# include <cstddef>

namespace ops {

template <typename Tp, typename Device>
struct relu_forward {
    /// @brief relu forward operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input  : the input array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, const Tp* input, size_t size);
};

template <typename Tp, typename Device>
struct relu_backward {
    /// @brief relu backward operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input  : the input array pointer
    /// @param grad   : the gradient array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, const Tp* input, const Tp* grad, size_t size);
};

} // namespace ops

#endif

