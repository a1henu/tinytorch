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
        .def(py::init<const tensor::Tensor<double>&>())
        .def("__deepcopy__", [](const tensor::Tensor<double>& self, py::dict) {
            return tensor::Tensor<double>(self);
        })
        // device
        .def("cpu", &tensor::Tensor<double>::cpu)
        .def("gpu", &tensor::Tensor<double>::gpu)
        .def("to_cpu", &tensor::Tensor<double>::to_cpu)
        .def("to_gpu", &tensor::Tensor<double>::to_gpu)
        .def("in_cpu", &tensor::Tensor<double>::in_cpu)
        .def("in_gpu", &tensor::Tensor<double>::in_gpu)
        .def("device", &tensor::Tensor<double>::device)
        // shape and dimension
        .def("dim", &tensor::Tensor<double>::dim)
        .def("shape", &tensor::Tensor<double>::get_shape)
        .def("reshape", py::overload_cast<const std::vector<int>&>(&tensor::Tensor<double>::reshape, py::const_))
        .def("transpose", &tensor::Tensor<double>::transpose)
        .def("size", &tensor::Tensor<double>::get_tol_size)
        .def("__len__", [] (const tensor::Tensor<double>& self) {
            return self.get_shape()[0];
        })
        // operators
        .def("power", &tensor::Tensor<double>::power)
        .def("ewise_mul", &tensor::Tensor<double>::ewise_mul)
        .def("__add__", &tensor::Tensor<double>::operator+)
        .def("__sub__", &tensor::Tensor<double>::operator-)
        .def("__matmul__", py::overload_cast<const tensor::Tensor<double>&>(&tensor::Tensor<double>::operator*, py::const_))
        .def("__mul__", py::overload_cast<const double>(&tensor::Tensor<double>::operator*, py::const_))
        .def("__rmul__", [](const tensor::Tensor<double>& self, double scalar) {
            return self * scalar;  
        })
        .def("__eq__", [](const tensor::Tensor<double>& self, py::object other) {
            if (!py::isinstance<tensor::Tensor<double>>(other)) {
                return false;
            }
            return self == other.cast<tensor::Tensor<double>>();
        })
        .def("__ne__", [](const tensor::Tensor<double>& self, py::object other) {
            if (!py::isinstance<tensor::Tensor<double>>(other)) {
                return true;
            }
            return self != other.cast<tensor::Tensor<double>>();
        })
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
        .def_static("zeros", &tensor::Tensor<double>::zeros)
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
        )
        .def("to_numpy", 
            [] (const tensor::Tensor<double>& self) {
                std::vector<int> shape = self.get_shape();
                std::vector<ssize_t> numpy_shape(shape.begin(), shape.end());
                size_t total_size = self.get_tol_size();
                
                py::array_t<double> result(numpy_shape);
                
                py::buffer_info buf = result.request();
                double* ptr = static_cast<double*>(buf.ptr);
                
                if (self.in_cpu()) {
                    std::memcpy(ptr, self.get_data(), total_size * sizeof(double));
                } else {
                    memory::copy_mem_op<double, device::CPU, device::GPU>()(
                        device::cpu_device, 
                        device::gpu_device, 
                        ptr, 
                        self.get_data(), 
                        total_size
                    );
                }
                
                return result;
            }, "Convert Tensor to numpy array");
}
