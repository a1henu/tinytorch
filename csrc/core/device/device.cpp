/**  
 * @file device.cpp
 * @brief Implementations of devices class for multi-device.
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"

namespace device {

bool BaseDevice::is_cpu() const { return false; }
bool BaseDevice::is_gpu() const { return false; }

bool CPU::is_cpu() const { return true; }
bool CPU::is_gpu() const { return false; }


bool GPU::is_cpu() const { return false; }
bool GPU::is_gpu() const { return true; }

} // namespace device