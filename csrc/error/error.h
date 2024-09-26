/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 */

#include <exception>

namespace error {

class DeviceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InvalidArgument : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace error