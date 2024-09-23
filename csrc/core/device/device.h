/**  
 * @file device.h
 * @brief This header defines the device classes.
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


} // namespace device

#endif