/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 */

#ifndef CSRC_ERROR_ERROR_H
#define CSRC_ERROR_ERROR_H

#include <exception>

namespace error {

class DeviceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InvalidArgumentError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class TypeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace error

#endif // ERROR_H