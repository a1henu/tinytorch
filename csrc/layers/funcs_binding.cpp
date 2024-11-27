/**
 * @file layers_binding.cpp
 * @brief Layers class binding with pybind11
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "layers/layers.h"
#include "tensor/operators/tensor_activation.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_libfuncs, m) {
    m.def("fc_forward", &layers::fc_forward<double>);
    m.def("fc_backward", &layers::fc_backward<double>);
    m.def("conv2d_forward", &layers::conv2d_forward<double>);
    m.def("conv2d_backward", &layers::conv2d_backward<double>);
    m.def("max_pool_forward", &layers::max_pool_forward<double>);
    m.def("max_pool_backward", &layers::max_pool_backward<double>);
    m.def("softmax_forward", &layers::softmax_forward<double>);
    m.def("cross_entropy_forward", &layers::cross_entropy_forward<double>);
    m.def("cross_entropy_backward", &layers::cross_entropy_backward<double>);
    m.def("mse_forward", &layers::mse_forward<double>);
    m.def("mse_backward", &layers::mse_backward<double>);

    m.def("relu_forward", &tensor::relu_forward<double>);
    m.def("relu_backward", &tensor::relu_backward<double>);
    m.def("sigmoid_forward", &tensor::sigmoid_forward<double>);
    m.def("sigmoid_backward", &tensor::sigmoid_backward<double>);
}