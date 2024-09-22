/**  
 * @file device.h
 * @brief This header defines the device classes.
 * 
 */

#ifndef CSRC_CORE_DEVICE_DEVICE_H
#define CSRC_CORE_DEVICE_DEVICE_H

#include <stdexcept>

namespace device {

struct BaseDevice{
    BaseDevice();
    virtual ~BaseDevice();

    virtual bool is_cpu();
    virtual bool is_gpu();
};


struct CPU: public BaseDevice {
    ~CPU() override;

    bool is_cpu() override;
    bool is_gpu() override;
};

struct GPU: public BaseDevice {
    ~GPU() override;

    bool is_cpu() override;
    bool is_gpu() override;
};

constexpr CPU* cpu_device {};
constexpr GPU* gpu_device {};

}

#endif