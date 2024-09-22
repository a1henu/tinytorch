/**
 * @file device.cpp
 * @brief The implementation of device.h
 * 
 */

#include "device.h"

device::BaseDevice::BaseDevice() {}

device::BaseDevice::~BaseDevice() {}

bool device::BaseDevice::is_cpu() {
    throw std::runtime_error("Not implemented");
}

bool device::BaseDevice::is_gpu() {
    throw std::runtime_error("Not implemented");
}

device::CPU::~CPU() {}

bool device::CPU::is_cpu(){
    return true;
}

bool device::CPU::is_gpu(){
    return false;
}

device::GPU::~GPU() {}

bool device::GPU::is_gpu(){
    return true;
}

bool device::GPU::is_cpu(){
    return false;
}