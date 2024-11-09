/**
 * @file ops.h
 * @brief Math operators declaration
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#ifndef CSRC_CORE_KERNELS_OPS_H
#define CSRC_CORE_KERNELS_OPS_H

#include "core/device/device.h"
#include "core/memory/memory.h"
#include "error/error.h"

#include <cblas.h>

namespace ops {

template <typename Tp, typename Device>
struct add_op {
    /// @brief add operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input1 : the input1 array pointer
    /// @param input2 : the input2 array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, const Tp* input1, const Tp* input2, size_t size);
};

template <typename Tp, typename Device>
struct sub_op {
    /// @brief sub operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input1 : the input1 array pointer
    /// @param input2 : the input2 array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, const Tp* input1, const Tp* input2, size_t size);
};

template <typename Tp, typename Device>
struct matmul_op {
    /// @brief matmul operator for multi-device
    ///        C = alpha * A * B + beta * C
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param transa : whether to transpose A
    /// @param transb : whether to transpose B
    /// @param m      : the number of rows of A
    /// @param n      : the number of columns of B
    /// @param k      : the number of columns of A
    /// @param alpha  : the scalar alpha
    /// @param A      : the input array A
    /// @param lda    : the leading dimension of A
    /// @param B      : the input array B
    /// @param ldb    : the leading dimension of B
    /// @param beta   : the scalar beta
    /// @param C      : the output array C
    /// @param ldc    : the leading dimension of C
    void operator()(
        Device* device,
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
    );
};

template <typename Tp, typename Device>
struct equal_op {
    /// @brief equal operator for multi-device
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param input1 : the input1 array pointer
    /// @param input2 : the input2 array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, bool* output, const Tp* input1, const Tp* input2, size_t size);
};

template <typename Tp, typename Device>
struct ones_op {
    /// @brief set all elements to 1
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param size   : the size of the array 
    void operator()(Device* device, Tp* output, size_t size);
};

template <typename Tp, typename Device>
struct eye_op {
    /// @brief set the diagonal elements to 1
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param output : the output array pointer
    /// @param dim    : the dimension of the square matrix
    void operator()(Device* device, Tp* output, size_t dim);
};

template <typename Tp, typename Device>
struct transpose_op {
    /// @brief transpose the matrix
    ///
    /// Inputs:
    /// @param device : the type of device
    /// @param input  : the input array pointer
    /// @param output : the output array pointer
    /// @param m      : the number of rows of the input matrix
    /// @param n      : the number of columns of the output matrix
    void operator()(Device* device, const Tp* input, Tp* output, const int m, const int n);
};

template <typename Tp, typename Device>
struct im2col_op {
    /// @brief convert the image to the column matrix
    ///
    /// Inputs:
    /// @param device    : the type of device
    /// @param data_im   : the input image array pointer
    /// @param data_col  : the output column matrix array pointer
    /// @param channels  : the number of channels of the image
    /// @param height    : the height of the image
    /// @param width     : the width of the image
    /// @param kernel_h  : the height of the kernel
    /// @param kernel_w  : the width of the kernel
    /// @param pad_h     : the height of the padding
    /// @param pad_w     : the width of the padding
    /// @param stride_h  : the height of the stride
    /// @param stride_w  : the width of the stride
    void operator()(
        Device* device,
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
    );
};

template <typename Tp, typename Device>
struct conv2d_forward_op {
    void operator()(
        Device* device,
        Tp* output,             // (batch_size, out_channels, height_out, width_out)
        const Tp* input,        // (batch_size, in_channels, height, width)
        const Tp* weight,       // (out_channels, in_channels, kernel_h, kernel_w)
        const Tp* bias,         // (out_channels)
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
    );
};



template <typename Tp, typename Device>
struct max_pool_forward_op {
    /// @brief max pooling forward operator
    ///
    /// Inputs:
    /// @param device    : the type of device
    /// @param img_out   : the output image array pointer
    /// @param mask_out  : the output mask array pointer
    /// @param img_in    : the input image array pointer
    /// @param batch_size : the batch size of the image
    /// @param channels  : the number of channels of the image
    /// @param height    : the height of the image
    /// @param width     : the width of the image
    /// @param kernel_h  : the height of the kernel
    /// @param kernel_w  : the width of the kernel
    /// @param pad_h     : the height of the padding
    /// @param pad_w     : the width of the padding
    /// @param stride_h  : the height of the stride
    /// @param stride_w  : the width of the stride
    void operator()(
        Device* device,
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
    );
};

template <typename Tp, typename Device>
struct max_pool_backward_op {
    /// @brief max pooling backward operator
    ///
    /// Inputs:
    /// @param device    : the type of device
    /// @param grad_im   : the input image array pointer
    /// @param mask_out  : the mask array pointer
    /// @param grad_out  : the gradient of the output image array pointer
    /// @param batch_size : the batch size of the image
    /// @param channels  : the number of channels of the image
    /// @param height    : the height of the image
    /// @param width     : the width of the image
    /// @param kernel_h  : the height of the kernel
    /// @param kernel_w  : the width of the kernel
    /// @param pad_h     : the height of the padding
    /// @param pad_w     : the width of the padding
    /// @param stride_h  : the height of the stride
    /// @param stride_w  : the width of the stride
    void operator()(
        Device* device,
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
    );
};

} // namespace ops

#endif