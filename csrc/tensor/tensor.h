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
    Tensor(const std::vector<int>& shape, const device::BaseDevice* device);

    Tensor(const std::vector<int>& shape, const device::BaseDevice* device, const std::vector<Tp>& data);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    ~Tensor();

    Tensor<Tp>& cpu();
    Tensor<Tp>& gpu();

    int get_tol_size() const;

private:
    std::vector<int> shape;
    device::BaseDevice* device;

    Tp* p_data;
};

}

#endif