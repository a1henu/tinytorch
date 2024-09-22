#ifndef CSRC_TENSOR_TENSOR_H
#define CSRC_TENSOR_TENSOR_H

#include <memory>
#include <stdexcept>
#include <vector>

#include "csrc/core/device/device.h"

namespace tensor {

template <typename Tp = double>
class Tensor {
public:
    Tensor(const std::vector<int>& shape, const DEVICE::BaseDevice* device);

    Tensor(const std::vector<int>& shape, const DEVICE::BaseDevice* device, const std::vector<Tp>& data);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    ~Tensor();

    Tensor<Tp>& cpu();
    Tensor<Tp>& gpu();

    int get_tol_size() const;

private:
    std::vector<int> shape;
    DEVICE::BaseDevice* device;

    Tp* p_data;
};

}

#endif