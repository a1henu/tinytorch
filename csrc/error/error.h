/**
 * @file error.h
 * @brief error types
 */

#ifndef CSRC_ERROR_ERROR_H
#define CSRC_ERROR_ERROR_H

#include <stdexcept>

namespace error {

class DeviceError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

class InvalidArgumentError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

class TypeError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

} // namespace error

#endif // ERROR_H