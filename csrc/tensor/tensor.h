/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_TENSOR_TENSOR_H
#define CSRC_TENSOR_TENSOR_H

#include <memory>
#include <stdexcept>
#include <vector>

#include "core/device/device.h"
#include "core/memory/memory.h"

namespace tensor {

template <typename Tp = double>
class Tensor {
public:
    Tensor();
    
    Tensor(const std::vector<int>& shape, device::BaseDevice* device);

    Tensor(
        const std::vector<int>& shape, 
        device::BaseDevice* device,
        Tp* data
    );

    Tensor(const Tensor& other);

    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other);

    ~Tensor();

    Tensor<Tp>& cpu();
    Tensor<Tp>& gpu();

    bool in_cpu() const;
    bool in_gpu() const;

    const std::vector<int> get_shape() const;

    const Tp* get_data() const;
    void set_data(const Tp* data, device::BaseDevice* device_d) const;

    size_t get_tol_size() const;

    // Tensor operators here
    // these operators are implemented in:
    // /tensor/operators/tensor_math_ops.cpp
    
    template <typename T>
    friend Tensor<T>& operator+(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    friend Tensor<T>& operator-(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    friend bool operator==(const Tensor<T>& a, const Tensor<T>& b);

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