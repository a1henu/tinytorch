/**  
 * @file device.h
 * @brief Devices class for multi-device.
 * 
 */

#ifndef CSRC_CORE_DEVICE_DEVICE_H
#define CSRC_CORE_DEVICE_DEVICE_H

#include <stdexcept>

namespace device {

struct BaseDevice {
    virtual bool is_cpu() const = 0;
    virtual bool is_gpu() const = 0;
};

struct CPU : public BaseDevice {
    bool is_cpu() const override;
    bool is_gpu() const override;
};

struct GPU : public BaseDevice {
    bool is_cpu() const override;
    bool is_gpu() const override;
};

constexpr CPU* cpu_device {};
constexpr GPU* gpu_device {};

} // namespace device

#endif