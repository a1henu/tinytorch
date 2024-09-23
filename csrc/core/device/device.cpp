/**  
 * @file device.cpp
 * @brief This implementations of the device classes.
 * 
 */

#include "core/device/device.h"

namespace device {

bool CPU::is_cpu() const { return true; }
bool CPU::is_gpu() const { return false; }


bool GPU::is_cpu() const { return false; }
bool GPU::is_gpu() const { return true; }

} // namespace device