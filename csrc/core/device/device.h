/**  
 * @file device.h
 * @brief This header defines the device classes.
 * 
 */

#ifndef CSRC_CORE_DEVICE_DEVICE_H
#define CSRC_CORE_DEVICE_DEVICE_H

#include <stdexcept>

namespace DEVICE {

struct BaseDevice{
    virtual bool is_cpu();
    virtual bool is_gpu();
};


struct CPU: public BaseDevice {
    bool is_cpu() override;
    bool is_gpu() override;
};

struct GPU: public BaseDevice {
    bool is_cpu() override;
    bool is_gpu() override;
};

constexpr DEVICE::CPU* cpu_device {};
constexpr DEVICE::GPU* gpu_device {};

}

#endif