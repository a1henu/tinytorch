/**
 * @file ops.cpp
 * @brief Math operators implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cblas.h>
#include <limits>
#include <cstring>

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
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        const int channels_col = channels * kernel_h * kernel_w;
        
        for (int c = 0; c < channels_col; ++c) {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / (kernel_h * kernel_w);
            
            for (int h = 0; h < height_out; ++h) {
                for (int w = 0; w < width_out; ++w) {
                    int h_pad = h * stride_h - pad_h + h_offset;
                    int w_pad = w * stride_w - pad_w + w_offset;
                    
                    data_col[(c * height_out + h) * width_out + w] = 
                        (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) ?
                        data_im[(c_im * height + h_pad) * width + w_pad] : 0;
                }
            }
        }
    }
};

template <typename Tp>
struct conv2d_forward_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* output,
        const Tp* input,
        const Tp* weight,
        const Tp* bias,
        const int batch_size,
        const int in_channels,
        const int out_channels,
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
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        const int col_size = in_channels * kernel_h * kernel_w * height_out * width_out;
        Tp* col = new Tp[col_size];

        // for each batch
        for (int b = 0; b < batch_size; ++b) {
            im2col_op<Tp, device::CPU>()(
                device,
                input + b * in_channels * height * width,
                col,
                in_channels,
                height,
                width,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w
            );

            matmul_op<Tp, device::CPU>()(
                device,
                "N", "N",
                out_channels,
                height_out * width_out,
                in_channels * kernel_h * kernel_w,
                1.0,
                weight,
                in_channels * kernel_h * kernel_w,
                col,
                height_out * width_out,
                0.0,
                output + b * out_channels * height_out * width_out,
                height_out * width_out
            );
            
            if (bias != nullptr) {
                for (int c = 0; c < out_channels; ++c) {
                    for (int hw = 0; hw < height_out * width_out; ++hw) {
                        output[b * out_channels * height_out * width_out + 
                               c * height_out * width_out + hw] += bias[c];
                    }
                }
            }
        }

        delete[] col;
    }
};

template <typename Tp>
struct max_pool_forward_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* img_out,
        int* mask_out,
        const Tp* img_in,
        const int batch_size,
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
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        // for each batch
        for (int b = 0; b < batch_size; ++b) {
            // for each channel
            for (int c = 0; c < channels; ++c) {
                const int batch_channel_offset = (b * channels + c);
                const Tp* img = img_in + batch_channel_offset * height * width;
                Tp* out = img_out + batch_channel_offset * height_out * width_out;
                int* mask = mask_out + batch_channel_offset * height_out * width_out;

                // for each point in the output img
                for (int i = 0; i < height_out; ++i) {
                    for (int j = 0; j < width_out; ++j) {
                        Tp max_val = img[(i * stride_h - pad_h) * width + j * stride_w - pad_w];
                        int max_idx = 0;

                        // for each point in the kernel
                        for (int ii = 0; ii < kernel_h; ++ii) {
                            for (int jj = 0; jj < kernel_w; ++jj) {
                                int h = i * stride_h + ii - pad_h;
                                int w = j * stride_w + jj - pad_w;

                                if (h >= 0 && h < height && w >= 0 && w < width) {
                                    Tp val = img[h * width + w];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = ii * kernel_w + jj;
                                    }
                                }
                            }
                        }

                        out[i * width_out + j] = max_val;
                        mask[i * width_out + j] = max_idx;
                    }
                }
            }
        }
    }
};

template <typename Tp>
struct max_pool_backward_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        Tp* grad_im,
        const int* mask_out,
        const Tp* grad_out,
        const int batch_size,
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
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        // set input img to zero
        memset(grad_im, 0, batch_size * channels * height * width * sizeof(Tp));

        // for each batch
        for (int b = 0; b < batch_size; ++b) {
            // for each channel
            for (int c = 0; c < channels; ++c) {
                const int batch_channel_offset = (b * channels + c);
                Tp* img = grad_im + batch_channel_offset * height * width;
                const Tp* grad = grad_out + batch_channel_offset * height_out * width_out;
                const int* mask = mask_out + batch_channel_offset * height_out * width_out;

                // for each point in the output img
                for (int i = 0; i < height_out; ++i) {
                    for (int j = 0; j < width_out; ++j) {
                        const int max_idx = mask[i * width_out + j];

                        const int ii = max_idx / kernel_w;
                        const int jj = max_idx % kernel_w;

                        const int h = i * stride_h + ii - pad_h;
                        const int w = j * stride_w + jj - pad_w;

                        img[h * width + w] += grad[i * width_out + j];
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

template <typename Tp>
struct conv2d_forward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        const Tp* input,
        const Tp* weight,
        const Tp* bias,
        const int batch_size,
        const int in_channels,
        const int height,
        const int width,
        const int out_channels,
        const int kernel_h,
        const int kernel_w,
        const int pad_h,
        const int pad_w,
        const int stride_h,
        const int stride_w
    ) {
        throw error::DeviceError("conv2d_forward_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct max_pool_forward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* img_out,
        int* mask_out,
        const Tp* img_in,
        const int batch_size,
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
        throw error::DeviceError("max_pool_forward_op<GPU> can not be called without CUDA support.");
    }
};

template <typename Tp>
struct max_pool_backward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* img_in,
        const int* mask_out,
        const Tp* grad_out,
        const int batch_size,
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
        throw error::DeviceError("max_pool_backward_op<GPU> can not be called without CUDA support.");
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

template struct conv2d_forward_op<int, device::GPU>;
template struct conv2d_forward_op<float, device::GPU>;
template struct conv2d_forward_op<double, device::GPU>;

template struct max_pool_forward_op<int, device::GPU>;
template struct max_pool_forward_op<float, device::GPU>;
template struct max_pool_forward_op<double, device::GPU>;

template struct max_pool_backward_op<int, device::GPU>;
template struct max_pool_backward_op<float, device::GPU>;
template struct max_pool_backward_op<double, device::GPU>;

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

template struct conv2d_forward_op<int, device::CPU>;
template struct conv2d_forward_op<float, device::CPU>;
template struct conv2d_forward_op<double, device::CPU>;

template struct max_pool_forward_op<int, device::CPU>;
template struct max_pool_forward_op<float, device::CPU>;
template struct max_pool_forward_op<double, device::CPU>;

template struct max_pool_backward_op<int, device::CPU>;
template struct max_pool_backward_op<float, device::CPU>;
template struct max_pool_backward_op<double, device::CPU>;

} // namespace ops