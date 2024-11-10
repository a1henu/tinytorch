/**
 * @file tensor_binding.cpp
 * @brief Tensor class binding with pybind11
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <sstream>

#include "tensor/tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_libtensor, m) {
    py::enum_<tensor::DeviceType>(m, "DeviceType")
        .value("CPU", tensor::DeviceType::CPU)
        .value("GPU", tensor::DeviceType::GPU)
        .export_values();

    py::class_<tensor::Tensor<double>>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, tensor::DeviceType>())
        .def(py::init<const std::vector<int>&, tensor::DeviceType, const std::vector<double>&>())
        .def("__copy__", [](const tensor::Tensor<double> &self) {
            return tensor::Tensor<double>(self);
        })
        .def("__deepcopy__", [](const tensor::Tensor<double> &self, py::dict) {
            return tensor::Tensor<double>(self);
        })
        // device
        .def("cpu", &tensor::Tensor<double>::cpu)
        .def("gpu", &tensor::Tensor<double>::gpu)
        .def("to_cpu", &tensor::Tensor<double>::to_cpu)
        .def("to_gpu", &tensor::Tensor<double>::to_gpu)
        .def("in_cpu", &tensor::Tensor<double>::in_cpu)
        .def("in_gpu", &tensor::Tensor<double>::in_gpu)
        // shape and dimension
        .def("dim", &tensor::Tensor<double>::dim)
        .def("shape", &tensor::Tensor<double>::get_shape)
        .def("reshape", py::overload_cast<const std::vector<int>&>(&tensor::Tensor<double>::reshape, py::const_))
        .def("transpose", &tensor::Tensor<double>::transpose)
        .def("size", &tensor::Tensor<double>::get_tol_size)
        // operators
        .def("__add__", &tensor::Tensor<double>::operator+)
        .def("__sub__", &tensor::Tensor<double>::operator-)
        .def("__matmul__", &tensor::Tensor<double>::operator*)
        .def("__eq__", &tensor::Tensor<double>::operator==)
        .def("__assign__", [](tensor::Tensor<double> &self, const tensor::Tensor<double> &other) -> tensor::Tensor<double>& {
            self = other;
            return self;
        }, py::is_operator())
        // built-in methods
        .def("__getitem__", [](const tensor::Tensor<double>& self, const std::vector<int>& indices) {
            return self[indices];
        })
        .def("__repr__", 
            [](const tensor::Tensor<double>& tensor) {
                std::ostringstream oss;
                oss << tensor;
                return oss.str();
            }
        )
        .def("__str__",     
            [](const tensor::Tensor<double>& tensor) {
                std::ostringstream oss;
                oss << tensor;
                return oss.str();
            }
        )
        // static methods
        .def_static("ones", py::overload_cast<const std::vector<int>&, tensor::DeviceType>(&tensor::Tensor<double>::ones))
        .def_static("from_numpy", 
            [] (py::array_t<double, py::array::c_style> array) {
                py::buffer_info buf = array.request();
                double* data = static_cast<double*>(buf.ptr);

                std::vector<int> shape;
                for (auto dim : buf.shape) {
                    shape.push_back(static_cast<int>(dim));
                }

                return tensor::Tensor<double>(shape, tensor::DeviceType::CPU, data);
            },
            "Create a Tensor from numpy array"
        );
}