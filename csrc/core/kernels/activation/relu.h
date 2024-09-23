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

namespace activation {

template <typename Tp, typename Device>
struct relu_forward {
    void operator()(Device* device, Tp* output, Tp* input, size_t size);
};

template <typename Tp, typename Device>
struct relu_backward {
    void operator()(Device* device, Tp* output, Tp* input, Tp* grad, size_t size);
};

} // namespace activation

#endif