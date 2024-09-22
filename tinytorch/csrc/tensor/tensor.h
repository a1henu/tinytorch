#ifndef CSRC_TENSOR_TENSOR_H
#define CSRC_TENSOR_TENSOR_H

#include <memory>
#include <stdexcept>
#include <vector>

#include "tinytorch/csrc/core/device/device.h"

namespace tinytorch {

class Tensor {

public:
    Tensor() {
        throw std::runtime_error("Not implemented yet");
    }

    Tensor(std::vector<int>& shape, DEVICE::BaseDevice* device) : 
        shape(shape), device(device) { }

    Tensor(std::vector<int>& shape, DEVICE::BaseDevice* device, std::shared_ptr<double>& p_data) : 
        shape(shape), device(device), p_data(p_data) { }

    Tensor(const Tensor& other) : 
        shape(other.shape), device(other.device), p_data(other.p_data) { }

    Tensor(Tensor&& other) : 
        shape(std::move(other.shape)), device(std::move(other.device)), p_data(std::move(other.p_data)) { }

    inline Tensor cpu() {
        return Tensor(shape, DEVICE::cpu_device, p_data);
    }

    inline Tensor cuda() {
        return Tensor(shape, DEVICE::cuda_device, p_data);
    }

private:
    std::vector<int> shape;
    DEVICE::BaseDevice* device;

    std::shared_ptr<double> p_data;
};

}

#endif