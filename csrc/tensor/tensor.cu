/**
 * @file tensor.cu
 * @brief Tensor class implementation
 */

#include <vector>

#include "tensor.h"

namespace tensor {

template<typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    const DEVICE::BaseDevice* device
) : shape(shape), device(device) {
    int tol_size = get_tol_size();
    if (device->is_cpu()) {
        p_data = new Tp[tol_size];
    } else if (device->is_gpu()) {
        cudaMalloc(&p_data, tol_size * sizeof(Tp));
    } else (
        throw std::runtime_error("Unknown device type")
    )
}

template<typename Tp>
Tensor<Tp>::Tensor(
    const std::vector<int>& shape, 
    const DEVICE::BaseDevice* device,
    const std::vector<Tp>& data
) : shape(shape), device(device) {
    int tol_size = get_tol_size();
    if (device->is_cpu()) {
        p_data = new Tp[tol_size];
        std::copy(data.begin(), data.begin() + std::min(tol_size, data.size()), p_data);
    } else if (device->is_gpu()) {
        Tp* tmp_data = new Tp[tol_size];
        std::copy(data.begin(), data.begin() + std::min(tol_size, data.size()), tmp_data);
        cudaMalloc(&p_data, tol_size * sizeof(Tp));
        cudaMemcpy(p_data, tmp_data, tol_size * sizeof(Tp), cudaMemcpyHostToDevice);
        delete[] tmp_data;
    } else (
        throw std::runtime_error("Unknown device type")
    )
}

template<typename Tp>
Tensor<Tp>::Tensor(Tensor<Tp>&& other) : shape(std::move(other.shape)) {
    device = other.device;
    p_data = other.p_data;
    other.p_data = nullptr;
}

template<typename Tp>
Tensor<Tp>::~Tensor() {
    if (device->is_gpu()) {
        cudaFree(p_data);
    } else if (device->is_cpu()) {
        delete[] p_data;
    } else {
        throw std::runtime_error("Unknown device type");
    }
}

template<typename Tp>
inline Tensor<Tp>& Tensor<Tp>::cpu() {
    if (device->is_cpu()) {
        return *this;
    } else if (device->is_gpu()) {
        int tol_size = get_tol_size();
        Tp* tmp_data = new Tp[tol_size];
        cudaMemcpy(tmp_data, p_data, tol_size * sizeof(Tp), cudaMemcpyDeviceToHost);
        cudaFree(p_data);
        p_data = tmp_data;
        device = DEVICE::cpu_device;
        return *this;
    } else {
        throw std::runtime_error("Unknown device type");
    }
}

template<typename Tp>
inline Tensor<Tp>& Tensor<Tp>::gpu() {
    if (device->is_gpu()) {
        return *this;
    } else if (device->is_cpu()) {
        int tol_size = get_tol_size();
        Tp* tmp_data = nullptr;
        cudaMalloc(&tmp_data, tol_size * sizeof(Tp));
        cudaMemcpy(tmp_data, p_data, tol_size * sizeof(Tp), cudaMemcpyHostToDevice);
        delete p_data;
        p_data = tmp_data;
        device = DEVICE::gpu_device;
        return *this
    } else {
        throw std::runtime_error("Unknown device type");
    }
}


template<typename Tp>
int Tensor<Tp>::get_tol_size() const {
    int s = 1;
    for (auto& i : shape) {
        s *= i;
    }
    return s;
}

}