/**
 * @file device.cpp
 * @brief The implementation of device.h
 * 
 */

#include "device.h"

bool DEVICE::BaseDevice::is_cpu(){
    return false;
}

bool DEVICE::BaseDevice::is_gpu(){
    return false;
}

bool DEVICE::CPU::is_cpu(){
    return true;
}

bool DEVICE::GPU::is_gpu(){
    return true;
}