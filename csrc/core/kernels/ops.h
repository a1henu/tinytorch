/**
 * @file ops.h
 * @brief Math operators declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_OPS_H
#define CSRC_CORE_KERNELS_OPS_H

#include "core/device/device.h"

namespace ops {

template <typename Tp, typename Device>
struct add_op {
    /// @brief add operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input1 : the input1 array pointer
    /// @param input2 : the input2 array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, Tp* input1, Tp* input2, size_t size);
};

template <typename Tp, typename Device>
struct sub_op {
    /// @brief sub operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input1 : the input1 array pointer
    /// @param input2 : the input2 array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, Tp* input1, Tp* input2, size_t size);
};

template <typename Tp, typename Device>
struct equal_op {
    /// @brief equal operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input1 : the input1 array pointer
    /// @param input2 : the input2 array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, bool* output, Tp* input1, Tp* input2, size_t size);
};

} // namespace ops

#endif