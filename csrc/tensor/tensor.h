/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_TENSOR_TENSOR_H
#define CSRC_TENSOR_TENSOR_H

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "core/device/device.h"
#include "core/memory/memory.h"

namespace tensor {

enum class DeviceType {
    CPU = 0,
    GPU = 1,
};

template <typename Tp = double>
class Tensor {
public:
    Tensor();
    
    Tensor(const std::vector<int>& shape, DeviceType device);

    Tensor(const std::initializer_list<int>& shape, DeviceType device);

    Tensor(
        const std::vector<int>& shape, 
        DeviceType device,
        Tp* data
    );

    Tensor(
        const std::initializer_list<int>& shape,
        DeviceType device,
        Tp* data
    );

    Tensor(const Tensor& other);

    Tensor& operator=(const Tensor& other);

    ~Tensor();

    Tensor<Tp>& cpu();
    Tensor<Tp>& gpu();

    bool in_cpu() const;
    bool in_gpu() const;

    const std::vector<int> get_shape() const;
    Tensor<Tp> reshape(const std::vector<int>& shape);
    Tensor<Tp> reshape(const std::initializer_list<int>& shape);

    const Tp* get_data() const;
    void set_data(Tp* data, size_t size, DeviceType device_d) const;

    size_t get_tol_size() const;

    // Tensor operators here
    // these operators are implemented in:
    // /tensor/operators/tensor_math_ops.cpp
    
    // template <typename T>
    // friend Tensor<T>& operator+(const Tensor<T>& a, const Tensor<T>& b);

    // template <typename T>
    // friend Tensor<T>& operator-(const Tensor<T>& a, const Tensor<T>& b);

    // template <typename T>
    // friend bool operator==(const Tensor<T>& a, const Tensor<T>& b);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    bool operator==(const Tensor& other) const;

    Tp& operator[](const std::vector<int>& indices) const;
    Tp& operator[](const std::initializer_list<int>& indices) const;

    static int get_index(const std::vector<int>& shape, const std::vector<int>& indices);

private:
    std::vector<int> shape;
    device::BaseDevice* device = nullptr;
    Tp* p_data = nullptr;

    using malloc_cpu_op = memory::malloc_mem_op<Tp, device::CPU>;
    using malloc_gpu_op = memory::malloc_mem_op<Tp, device::GPU>;

    using free_cpu_op = memory::free_mem_op<Tp, device::CPU>;
    using free_gpu_op = memory::free_mem_op<Tp, device::GPU>;

    using copy_c2c_op = memory::copy_mem_op<Tp, device::CPU, device::CPU>;
    using copy_c2g_op = memory::copy_mem_op<Tp, device::GPU, device::CPU>;
    using copy_g2c_op = memory::copy_mem_op<Tp, device::CPU, device::GPU>;
    using copy_g2g_op = memory::copy_mem_op<Tp, device::GPU, device::GPU>;

    using set_cpu_op = memory::set_mem_op<Tp, device::CPU>;
    using set_gpu_op = memory::set_mem_op<Tp, device::GPU>;
};

}

#endif