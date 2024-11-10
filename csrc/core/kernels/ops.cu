/**
 * @file ops.cu
 * @brief Math operators implementation for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <cublas_v2.h>

#include "core/kernels/ops.h"

#include "macros.h"

namespace ops {

template <typename Tp>
__global__ void
kernel_add(Tp* output, const Tp* input1, const Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input1[i] + input2[i];
    }
}

template <typename Tp>
__global__ void
kernel_sub(Tp* output, const Tp* input1, const Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input1[i] - input2[i];
    }
}

__global__ void
assign_to_true(bool* flag) {
    *flag = true;
}

template <typename Tp>
__global__ void
kernel_eq(bool* output, const Tp* input1, const Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        if (input1[i] != input2[i]) {
            *output = false;
        }
    }
}

template <typename Tp>
__global__ void
kernel_ones(Tp* arr, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        arr[i] = 1;
    }
}

template <typename Tp>
__global__ void
kernel_eye(Tp* arr, size_t dim) {
    CUDA_KERNEL_LOOP(i, dim) {
        arr[i * dim + i] = 1;
    }
}

template <typename Tp>
__global__ void
kernel_trans(const Tp* input, Tp* output, const int m, const int n) {
    const int i = threadIdx.x / n;
    const int j = threadIdx.x % n;
    if (i < m) {
        output[j * m + i] = input[i * n + j];
    }
}

template <typename Tp>
__global__ void
kernel_im2col(
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
    const int stride_w,
    const int height_out,
    const int width_out
) {
    const int channels_col = channels * kernel_h * kernel_w;
    CUDA_KERNEL_LOOP(index, channels_col * height_out * width_out) {
        int w = index % width_out;
        int h = (index / width_out) % height_out;
        int c = index / (height_out * width_out);
        
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);
        
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        
        data_col[index] = 
            (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) ?
            data_im[(c_im * height + h_pad) * width + w_pad] : 0;
    }
}

template <typename Tp>
__global__ void
kernel_add_bias(
    Tp* output,
    const Tp* bias,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    const int size = batch_size * channels * height * width;
    CUDA_KERNEL_LOOP(index, size) {
        const int c = (index / (height * width)) % channels;
        output[index] += bias[c];
    }
}

template <typename Tp>
__global__ void
kernel_conv2d_forward(
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
    const int stride_w,
    const int height_out,
    const int width_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) return;
    
    const int w_out = idx % width_out;
    const int h_out = (idx / width_out) % height_out;
    const int c_out = (idx / (width_out * height_out)) % out_channels;
    const int b = idx / (width_out * height_out * out_channels);
    
    Tp sum = 0;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_in = h_out * stride_h - pad_h + kh;
                const int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                    const int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[idx] = sum;
}

template <typename Tp>
__global__ void
kernel_max_pool(
    Tp* output,
    int* mask,
    const Tp* data_im,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int height_out,
    const int width_out
) { 
    // for each channel
    CUDA_KERNEL_LOOP(index, batch_size * channels * height_out * width_out) {
        // calculate the index of the current channel
        const int b = index / (channels * height_out * width_out);
        const int c = (index / (height_out * width_out)) % channels;
        const int i = (index / width_out) % height_out;
        const int j = index % width_out;
        
        // get the pointer of the current channel
        const Tp* img = data_im + c * height * width;
        Tp* out = output + b * channels * height_out * width_out + c * height_out * width_out;
        int* mask_out = mask + b * channels * height_out * width_out + c * height_out * width_out;
        
        // find the max value in the kernel window
        Tp max_val = img[(i * stride_h - pad_h) * width + j * stride_w - pad_w];
        int max_idx = 0;
        
        // iterate over the kernel window
        for (int ii = 0; ii < kernel_h; ++ii) {
            for (int jj = 0; jj < kernel_w; ++jj) {
                const int h = i * stride_h + ii - pad_h;
                const int w = j * stride_w + jj - pad_w;
                
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
        mask_out[i * width_out + j] = max_idx;
    }
}

template <typename Tp>
__device__ void atomicAddWrapper(Tp* address, Tp val) {
    atomicAdd(address, val);
}

template <>
__device__ void atomicAddWrapper<double>(double* address, double val) {
    #if __CUDA_ARCH__ >= 600
        atomicAdd(address, val);
    #else
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    #endif
}

template <typename Tp>
__global__ void
kernel_max_pool_backward(
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
    const int stride_w,
    const int height_out,
    const int width_out
) {
    CUDA_KERNEL_LOOP(index, batch_size * channels * height_out * width_out) {
        const int w = index % width_out;
        const int h = (index / width_out) % height_out;
        const int c = (index / (height_out * width_out)) % channels;
        const int b = index / (channels * height_out * width_out);

        const int out_idx = b * channels * height_out * width_out + 
                            c * height_out * width_out + 
                            h * width_out + w;
        
        const int max_idx = mask_out[out_idx];
        const int h_im = h * stride_h - pad_h + max_idx / kernel_w;
        const int w_im = w * stride_w - pad_w + max_idx % kernel_w;

        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
            const int im_idx = b * channels * height * width +
                               c * height * width +
                               h_im * width + w_im;
            atomicAddWrapper(&grad_im[im_idx], grad_out[out_idx]);
        }
    }
}

template <typename Tp>
struct add_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        kernel_add<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input1, input2, size);
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
        kernel_sub<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input1, input2, size);
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
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasOperation_t transa_ = (*transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transb_ = (*transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
        if (std::is_same<Tp, float>::value) {
            cublasSgemm(
                handle,
                transb_,
                transa_,
                n,
                m,
                k,
                reinterpret_cast<const float*>(&alpha),
                reinterpret_cast<const float*>(B),
                ldb,
                reinterpret_cast<const float*>(A),
                lda,
                reinterpret_cast<const float*>(&beta),
                reinterpret_cast<float*>(C),
                ldc
            );
            cublasDestroy(handle);
        } else if (std::is_same<Tp, double>::value) {
            cublasDgemm(
                handle,
                transb_,
                transa_,
                n,
                m,
                k,
                reinterpret_cast<const double*>(&alpha),
                reinterpret_cast<const double*>(B),
                ldb,
                reinterpret_cast<const double*>(A),
                lda,
                reinterpret_cast<const double*>(&beta),
                reinterpret_cast<double*>(C),
                ldc
            );
            cublasDestroy(handle);
        } else {
            cublasDestroy(handle);
            throw std::runtime_error("Unsupported data type for matmul");
        }
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
        assign_to_true<<<1, 1>>>(output);
        kernel_eq<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input1, input2, size);
    }
};

template <typename Tp>
struct ones_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* arr, 
        size_t size
    ) {
        kernel_ones<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(arr, size);
    }
};

template <typename Tp>
struct eye_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* arr, 
        size_t dim
    ) {
        kernel_eye<Tp><<<CUDA_GET_BLOCKS(dim), CUDA_K_THREADS>>>(arr, dim);
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
        kernel_trans<Tp><<<CUDA_GET_BLOCKS(m * n), CUDA_K_THREADS>>>(input, output, m, n);
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
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        const int channels_col = channels * kernel_h * kernel_w;
        
        kernel_im2col<Tp><<<CUDA_GET_BLOCKS(channels_col * height_out * width_out), CUDA_K_THREADS>>>(
            data_im,
            data_col,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            height_out,
            width_out
        );
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
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        
        const int total_threads = batch_size * out_channels * height_out * width_out;
        kernel_conv2d_forward<Tp><<<CUDA_GET_BLOCKS(total_threads), CUDA_K_THREADS>>>(
            output,
            input,
            weight,
            bias,
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            height_out,
            width_out
        );
    }
};

template <typename Tp>
struct max_pool_forward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        int* mask,
        const Tp* data_im,
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

        kernel_max_pool<Tp><<<CUDA_GET_BLOCKS(batch_size * channels * height_out * width_out), CUDA_K_THREADS>>>(
            output,
            mask,
            data_im,
            batch_size,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            height_out,
            width_out
        );
    }
};

template <typename Tp>
struct max_pool_backward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
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
        // initialize the gradient to zero
        cudaMemset(grad_im, 0, batch_size * channels * height * width * sizeof(Tp));

        // calculate the size of the output img
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        // launch the kernel
        kernel_max_pool_backward<Tp><<<CUDA_GET_BLOCKS(batch_size * channels * height_out * width_out), CUDA_K_THREADS>>>(
            grad_im,
            mask_out,
            grad_out,
            batch_size,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            height_out,
            width_out
        );
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

} // namespace ops
