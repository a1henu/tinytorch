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

namespace tensor {

template <typename Tp = double>
class Tensor {
public:
    Tensor(const std::vector<int>& shape, device::BaseDevice* device);

    Tensor(
        const std::vector<int>& shape, 
        Tp* data, 
        device::BaseDevice* t_device, 
        device::BaseDevice* d_device
    );

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    ~Tensor();

    Tensor<Tp>& cpu();
    Tensor<Tp>& gpu();

    bool in_cpu() const;
    bool in_gpu() const;

    const std::vector<int> get_shape() const;
    const Tp* get_data() const;

    size_t get_tol_size() const;

private:
    std::vector<int> shape;
    device::BaseDevice* device;
    Tp* p_data;

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