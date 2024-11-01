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
                CblasRowMajor,
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
                CblasRowMajor,
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

template <typename Tp>
struct transpose_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        const Tp* input,
        Tp* output,
        const int m,
        const int n
    ) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                output[j * m + i] = input[i * n + j];
            }
        }
    }
};

template <typename Tp>
struct im2col_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        const Tp* data_im,
        Tp* data_col,
        const int channels,
        const int height,
        const int width,
        const int kernel_h,
        const int kernel_w,
        const int pad_h,
        const int pad_w,
        const int stride_h,
        const int stride_w
    ) {
        // calculate the size of the output img
        int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        // calculate the size of the col matrix(row-major) for each channel
        int width_col = kernel_h * kernel_w;
        int height_col = height_out * width_out;

        // for each channel
        for (int c = 0; c < channels; ++c) {
            const Tp* img = data_im + c * height * width;
            Tp* col = data_col + c * width_col * height_col;

            // for each point in the output col matrix
            for (int j = 0; j < width_col; ++j) {
                for (int i = 0; i < height_col; ++i) {
                    int h_offset = i / width_out * stride_h - pad_h;
                    int w_offset = i % width_out * stride_w - pad_w;

                    int kh_offset = j / kernel_w;
                    int kw_offset = j % kernel_w;

                    if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width) {
                        col[i + j * height_col] = img[h_offset + kh_offset + (w_offset + kw_offset) * height];
                    } else {
                        col[i + j * height_col] = 0;
                    }
                }
            }
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

template <typename Tp>
struct transpose_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        const Tp* input,
        Tp* output,
        const int m,
        const int n
    ) {
        throw error::DeviceError("transpose_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct im2col_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        const Tp* data_im,
        Tp* data_col,
        const int channels,
        const int height,
        const int width,
        const int kernel_h,
        const int kernel_w,
        const int pad_h,
        const int pad_w,
        const int stride_h,
        const int stride_w
    ) {
        throw error::DeviceError("im2col_op<GPU> can not be called without CUDA support.");
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

template struct transpose_op<int, device::GPU>;
template struct transpose_op<float, device::GPU>;
template struct transpose_op<double, device::GPU>;

template struct im2col_op<int, device::GPU>;
template struct im2col_op<float, device::GPU>;
template struct im2col_op<double, device::GPU>;

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

template struct transpose_op<int, device::CPU>;
template struct transpose_op<float, device::CPU>;
template struct transpose_op<double, device::CPU>;

template struct im2col_op<int, device::CPU>;
template struct im2col_op<float, device::CPU>;
template struct im2col_op<double, device::CPU>;

} // namespace ops