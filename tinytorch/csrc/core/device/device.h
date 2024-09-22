#ifndef CSRC_CORE_DEVICE_DEVICE_H
#define CSRC_CORE_DEVICE_DEVICE_H

#include <stdexcept>

namespace DEVICE {

struct BaseDevice{
    BaseDevice() {
        throw std::runtime_error("Not implemented yet");
    }
};


struct CPU: public BaseDevice {
    CPU() = default;
};

struct CUDA: public BaseDevice {
    CUDA() = default;
};

constexpr DEVICE::CPU* cpu_device;
constexpr DEVICE::CUDA* cuda_device;

}

#endif