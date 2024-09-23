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
    void operator()(Device* device, Tp* output, Tp* input, size_t size);
};

template <typename Tp, typename Device>
struct sigmoid_backward {
    void operator()(Device* device, Tp* output, Tp* input, Tp* grad, size_t size);
};

} // namespace activation

#endif