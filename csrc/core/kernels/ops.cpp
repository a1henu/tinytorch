/**
 * @file ops.cpp
 * @brief Math operators implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cblas.h>

#include "core/device/device.h"
#include "core/kernels/ops.h"

#include "error/error.h"

namespace ops {

template <typename Tp>
struct add_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = input1[i] + input2[i];
        }
    }
};

template <typename Tp>
struct sub_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = input1[i] - input2[i];
        }
    }
};

template <typename Tp>
struct matmul_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        const char* transa, 
        const char* transb,
        const int m,
        const int n,
        const int k,
        const Tp alpha,
        const Tp* A,
        const int lda,
        const Tp* B,
        const int ldb,
        const Tp beta,
        Tp* C,
        const int ldc
    ) {
        CBLAS_TRANSPOSE transa_ = transa[0] == 'N' ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE transb_ = transb[0] == 'N' ? CblasNoTrans : CblasTrans;
        if (std::is_same<Tp, float>::value) {
            cblas_sgemm(
                CblasColMajor,
                transa_,
                transb_,
                m,
                n,
                k,
                static_cast<float>(alpha),
                reinterpret_cast<const float*>(A),
                lda,
                reinterpret_cast<const float*>(B),
                ldb,
                static_cast<float>(beta),
                reinterpret_cast<float*>(C),
                ldc
            );
        } else if (std::is_same<Tp, double>::value) {
            cblas_dgemm(
                CblasColMajor,
                transa_,
                transb_,
                m,
                n,
                k,
                static_cast<double>(alpha),
                reinterpret_cast<const double*>(A),
                lda,
                reinterpret_cast<const double*>(B),
                ldb,
                static_cast<double>(beta),
                reinterpret_cast<double*>(C),
                ldc
            );
        } else {
            throw error::TypeError("matmul_op only supports float and double.");
        }
    }
};

template <typename Tp>
struct equal_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        bool* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        *output = true;
        for (int i = 0; i < size; ++i) {
            if (input1[i] != input2[i]) {
                *output = false;
                break;
            }
        }
    }
};

template <typename Tp>
struct ones_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        size_t size
    ) {
        for (int i = 0; i < size; ++i) {
            output[i] = 1;
        }
    }
};

template <typename Tp>
struct eye_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device, 
        Tp* output, 
        size_t dim
    ) {
        for (int i = 0; i < dim; ++i) {
            output[i + i * dim] = 1;
        }
    }
};

#ifndef __CUDA

template <typename Tp>
struct add_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        throw error::DeviceError("add_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct sub_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        throw error::DeviceError("sub_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct matmul_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        const char* transa, 
        const char* transb,
        const int m,
        const int n,
        const int k,
        const Tp alpha,
        const Tp* A,
        const int lda,
        const Tp* B,
        const int ldb,
        const Tp beta,
        Tp* C,
        const int ldc
    ) {
        throw error::DeviceError("matmul_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct equal_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        bool* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        throw error::DeviceError("equal_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct ones_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        size_t size
    ) {
        throw error::DeviceError("ones_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct eye_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        size_t dim
    ) {
        throw error::DeviceError("eye_op<GPU> can not be called without CUDA support.");
    }
};

template struct add_op<int, device::GPU>;
template struct add_op<float, device::GPU>;
template struct add_op<double, device::GPU>;

template struct sub_op<int, device::GPU>;
template struct sub_op<float, device::GPU>;
template struct sub_op<double, device::GPU>;

template struct matmul_op<int, device::GPU>;
template struct matmul_op<float, device::GPU>;
template struct matmul_op<double, device::GPU>;

template struct equal_op<int, device::GPU>;
template struct equal_op<float, device::GPU>;
template struct equal_op<double, device::GPU>;

template struct ones_op<int, device::GPU>;
template struct ones_op<float, device::GPU>;
template struct ones_op<double, device::GPU>;

template struct eye_op<int, device::GPU>;
template struct eye_op<float, device::GPU>;
template struct eye_op<double, device::GPU>;

#endif

template struct add_op<int, device::CPU>;
template struct add_op<float, device::CPU>;
template struct add_op<double, device::CPU>;

template struct sub_op<int, device::CPU>;
template struct sub_op<float, device::CPU>;
template struct sub_op<double, device::CPU>;

template struct matmul_op<int, device::CPU>;
template struct matmul_op<float, device::CPU>;
template struct matmul_op<double, device::CPU>;

template struct equal_op<int, device::CPU>;
template struct equal_op<float, device::CPU>;
template struct equal_op<double, device::CPU>;

template struct ones_op<int, device::CPU>;
template struct ones_op<float, device::CPU>;
template struct ones_op<double, device::CPU>;

template struct eye_op<int, device::CPU>;
template struct eye_op<float, device::CPU>;
template struct eye_op<double, device::CPU>;

} // namespace ops