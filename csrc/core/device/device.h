/**  
 * @file device.h
 * @brief Devices class for multi-device.
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_DEVICE_DEVICE_H
#define CSRC_CORE_DEVICE_DEVICE_H

#include <stdexcept>

namespace device {

struct BaseDevice {
    virtual bool is_cpu() const;
    virtual bool is_gpu() const;
};

struct CPU : public BaseDevice {
    bool is_cpu() const override;
    bool is_gpu() const override;
};

struct GPU : public BaseDevice {
    bool is_cpu() const override;
    bool is_gpu() const override;
};

constexpr CPU* cpu_device = {};
constexpr GPU* gpu_device = {};

} // namespace device

#endif