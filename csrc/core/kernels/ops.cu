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

#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace ops {

template <typename Tp>
__global__ void compute_grad_bias_kernel(
    Tp* grad_bias,             // (C)
    const Tp* grad_output,     // (N, C, H, W)
    const int batch_size,      // N
    const int out_channels,    // C
    const int height_out,      // H
    const int width_out        // W
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= out_channels) return;
    
    Tp sum = 0.0;
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < height_out; ++h) {
            for (int w = 0; w < width_out; ++w) {
                int idx = ((n * out_channels + c) * height_out + h) * width_out + w;
                sum += grad_output[idx];
            }
        }
    }

    grad_bias[c] = sum;
}

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

template <typename Tp>
__global__ void
kernel_mul(Tp* output, const Tp* input, const Tp* num, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input[i] * *num;
    }
}

template <typename Tp>
__global__ void
kernel_ewise_mul(Tp* output, const Tp* input1, const Tp* input2, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = input1[i] * input2[i];
    }
}

template <typename Tp>
__global__ void
kernel_pow(Tp* output, const Tp* input, const Tp* num, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = powf(input[i], *num);
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
kernel_zeros(Tp* arr, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        arr[i] = 0;
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
kernel_bias_fc(Tp* output, const Tp* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int i = idx % out_features;
        output[idx] += bias[i];
    }
}

template <typename Tp>
__global__ void 
kernel_calc_db(Tp* grad_bias, const Tp* grad_output, int batch_size, int out_features) {
    CUDA_KERNEL_LOOP(i, out_features) {
        Tp sum = 0;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad_output[b * out_features + i];
        }
        grad_bias[i] = sum;
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
__global__ void kernel_col2im(
    const Tp* data_col,
    Tp* data_im,
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
        // parse the position in the col
        int w = index % width_out;
        int h = (index / width_out) % height_out;
        int c = index / (height_out * width_out);
        
        // calculate the offset in the kernel
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);
        
        // calculate the position in the original image
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
            atomicAdd(&data_im[(c_im * height + h_pad) * width + w_pad], data_col[index]);
        }
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
    CUDA_KERNEL_LOOP(index, batch_size * channels * height * width) {
        const int b = index / (channels * height * width);
        const int c = (index / (height * width)) % channels;
        const int h = (index / width) % height;
        const int w = index % width;
        output[
            b * channels * height * width +
            c * height * width +
            h * width + w
        ] += bias[c];
    }
}

template <typename Tp>
__global__ void
kernel_max_pool(
    Tp* img_out,
    Tp* mask_out,
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
        
        if (i < height_out && j < width_out) {
            const int batch_channel_offset = (b * channels + c);
            const Tp* img = img_in + batch_channel_offset * height * width;
            Tp* out = img_out + batch_channel_offset * height_out * width_out;
            Tp* mask = mask_out + batch_channel_offset * height_out * width_out;

            Tp max_val = img[(i * stride_h - pad_h) * width + j * stride_w - pad_w];
            int max_idx = 0;

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

template <typename Tp>
__global__ void
kernel_max_pool_backward(
    Tp* grad_im,
    const Tp* mask_out,
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
            atomicAdd(&grad_im[im_idx], grad_out[out_idx]);
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
struct mul_op<Tp, device::GPU> {
    void operator()(device::GPU* device, Tp* output, const Tp* arr, const Tp num, size_t size) {
        Tp* d_num;
        cudaMalloc(&d_num, sizeof(Tp));
        cudaMemcpy(d_num, &num, sizeof(Tp), cudaMemcpyHostToDevice);
        kernel_mul<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, arr, d_num, size);
        cudaFree(d_num);
    }
};

template <typename Tp>
struct ewise_mul_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device, 
        Tp* output, 
        const Tp* input1, 
        const Tp* input2, 
        size_t size
    ) {
        kernel_ewise_mul<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, input1, input2, size);
    }
};


template <typename Tp>
struct pow_op<Tp, device::GPU> {
    void operator()(device::GPU* device, Tp* output, const Tp* arr, const double num, size_t size) {
        Tp* d_num;
        cudaMalloc(&d_num, sizeof(Tp));
        cudaMemcpy(d_num, &num, sizeof(Tp), cudaMemcpyHostToDevice);
        kernel_pow<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, arr, d_num, size);
        cudaFree(d_num);
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
        if constexpr (std::is_same<Tp, float>::value) {
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
        } else if constexpr (std::is_same<Tp, double>::value) {
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
struct zeros_op<Tp, device::GPU> {
    void operator()(device::GPU* device, Tp* output, size_t size) {
        kernel_zeros<Tp><<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(output, size);
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
struct fc_forward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        const Tp* input,
        const Tp* weight,
        const Tp* bias,
        const int batch_size,
        const int in_features,
        const int out_features
    ) {
        if constexpr (std::is_same<Tp, float>::value) {
            matmul_op<float, device::GPU>()(
                device::gpu_device,
                "N",
                "N",
                batch_size,
                out_features,
                in_features,
                static_cast<float>(1.0),
                reinterpret_cast<const float*>(input),
                in_features,
                reinterpret_cast<const float*>(weight),
                out_features,
                static_cast<float>(0.0),
                reinterpret_cast<float*>(output),
                out_features
            );
        } else if constexpr (std::is_same<Tp, double>::value) {
            matmul_op<double, device::GPU>()(
                device::gpu_device,
                "N",
                "N",
                batch_size,
                out_features,
                in_features,
                static_cast<double>(1.0),
                reinterpret_cast<const double*>(input),
                in_features,
                reinterpret_cast<const double*>(weight),
                out_features,
                static_cast<double>(0.0),
                reinterpret_cast<double*>(output),
                out_features
            );
        } else {
            throw std::runtime_error("fc_forward_op only supports float and double.");
        }
        kernel_bias_fc<Tp><<<CUDA_GET_BLOCKS(batch_size * out_features), CUDA_K_THREADS>>>(output, bias, batch_size, out_features);
    }
};

template <typename Tp>
struct fc_backward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* grad_input,
        Tp* grad_weight,
        Tp* grad_bias,
        const Tp* grad_output,
        const Tp* input,
        const Tp* weight,
        const int batch_size,
        const int in_features,
        const int out_features
    ) {
        if constexpr (std::is_same<Tp, float>::value) {
            // Compute grad_input: dX = dY * W^T
            matmul_op<float, device::GPU>()(
                device::gpu_device,
                "N",
                "T",
                batch_size,
                in_features,
                out_features,
                static_cast<float>(1.0),
                reinterpret_cast<const float*>(grad_output),
                out_features,
                reinterpret_cast<const float*>(weight),
                out_features,
                static_cast<float>(0.0),
                reinterpret_cast<float*>(grad_input),
                in_features
            );

            // Compute grad_weight: dW = X^T * dY
            matmul_op<float, device::GPU>()(
                device::gpu_device,
                "T",
                "N",
                in_features,
                out_features,
                batch_size,
                static_cast<float>(1.0),
                reinterpret_cast<const float*>(input),
                in_features,
                reinterpret_cast<const float*>(grad_output),
                out_features,
                static_cast<float>(0.0),
                reinterpret_cast<float*>(grad_weight),
                out_features
            );

            // Compute grad_bias: db = sum(dY)
            kernel_calc_db<<<CUDA_GET_BLOCKS(out_features), CUDA_K_THREADS>>>(grad_bias, grad_output, batch_size, out_features);
        } else if constexpr (std::is_same<Tp, double>::value) {
            // Compute grad_input: dX = dY * W^T
            matmul_op<double, device::GPU>()(
                device::gpu_device,
                "N",
                "T",
                batch_size,
                in_features,
                out_features,
                static_cast<double>(1.0),
                reinterpret_cast<const double*>(grad_output),
                out_features,
                reinterpret_cast<const double*>(weight),
                out_features,
                static_cast<double>(0.0),
                reinterpret_cast<double*>(grad_input),
                in_features
            );

            // Compute grad_weight: dW = X^T * dY
            matmul_op<double, device::GPU>()(
                device::gpu_device,
                "T",
                "N",
                in_features,
                out_features,
                batch_size,
                static_cast<double>(1.0),
                reinterpret_cast<const double*>(input),
                in_features,
                reinterpret_cast<const double*>(grad_output),
                out_features,
                static_cast<double>(0.0),
                reinterpret_cast<double*>(grad_weight),
                out_features
            );

            // Compute grad_bias: db = sum(dY)
            kernel_calc_db<<<CUDA_GET_BLOCKS(out_features), CUDA_K_THREADS>>>(grad_bias, grad_output, batch_size, out_features);
        } else {
            throw std::runtime_error("fc_backward_op only supports float and double.");
        }
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
struct col2im_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        const Tp* data_col,
        Tp* data_im,
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

        kernel_col2im<Tp><<<CUDA_GET_BLOCKS(channels * height_out * width_out), CUDA_K_THREADS>>>(
            data_col,
            data_im,
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
        // calculate the size of the output img
        const int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        const int col_size = in_channels * kernel_h * kernel_w * height_out * width_out;
        Tp* col;
        cudaMalloc(&col, col_size * sizeof(Tp));

        // for each batch
        for (int b = 0; b < batch_size; ++b) {
            im2col_op<Tp, device::GPU>()(
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

            matmul_op<Tp, device::GPU>()(
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
        }

        kernel_add_bias<Tp><<<CUDA_GET_BLOCKS(batch_size * out_channels * height_out * width_out), CUDA_K_THREADS>>>(
            output,
            bias,
            batch_size,
            out_channels,
            height_out,
            width_out
        );

        cudaFree(col);
    }
};

template <typename Tp>
struct conv2d_backward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* grad_input,
        Tp* grad_weight,
        Tp* grad_bias,
        const Tp* grad_output,
        const Tp* input,
        const Tp* weight,
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
        const int channels_col = in_channels * kernel_h * kernel_w;
        
        Tp* col;
        Tp* grad_col;
        cudaMalloc(&col, sizeof(Tp) * channels_col * height_out * width_out);
        cudaMalloc(&grad_col, sizeof(Tp) * channels_col * height_out * width_out);
        
        cudaMemset(grad_bias, 0, sizeof(Tp) * out_channels);
        compute_grad_bias_kernel<Tp><<<CUDA_GET_BLOCKS(out_channels), CUDA_K_THREADS>>>(
            grad_bias,
            grad_output,
            batch_size,
            out_channels,
            height_out,
            width_out
        );
        cudaDeviceSynchronize();

        cudaMemset(grad_weight, 0, sizeof(Tp) * out_channels * channels_col);
        for (int b = 0; b < batch_size; ++b) {
            im2col_op<Tp, device::GPU>()(
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
            
            // grad_weight += grad_output * col^T
            matmul_op<Tp, device::GPU>()(
                device,
                "N", "T",
                out_channels,
                channels_col,
                height_out * width_out,
                1.0,
                grad_output + b * out_channels * height_out * width_out,
                height_out * width_out,
                col,
                height_out * width_out,
                1.0,
                grad_weight,
                channels_col
            );
        }
        
        cudaMemset(grad_input, 0, sizeof(Tp) * batch_size * in_channels * height * width);
        for (int b = 0; b < batch_size; ++b) {
            // weight^T * grad_output
            matmul_op<Tp, device::GPU>()(
                device,
                "T", "N",
                channels_col,
                height_out * width_out,
                out_channels,
                1.0,
                weight,
                channels_col,
                grad_output + b * out_channels * height_out * width_out,
                height_out * width_out,
                0.0,
                grad_col,
                height_out * width_out
            );
            
            col2im_op<Tp, device::GPU>()(
                device,
                grad_col,
                grad_input + b * in_channels * height * width,
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
        }
        
        cudaFree(col);
        cudaFree(grad_col);
    }
};

template <typename Tp>
struct max_pool_forward_op<Tp, device::GPU> {
    void operator()(
        device::GPU* device,
        Tp* output,
        Tp* mask,
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
        const Tp* mask_out,
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

template struct mul_op<int, device::GPU>;
template struct mul_op<float, device::GPU>;
template struct mul_op<double, device::GPU>;

template struct ewise_mul_op<int, device::GPU>;
template struct ewise_mul_op<float, device::GPU>;
template struct ewise_mul_op<double, device::GPU>;

template struct pow_op<int, device::GPU>;
template struct pow_op<float, device::GPU>;
template struct pow_op<double, device::GPU>;

template struct matmul_op<int, device::GPU>;
template struct matmul_op<float, device::GPU>;
template struct matmul_op<double, device::GPU>;

template struct equal_op<int, device::GPU>;
template struct equal_op<float, device::GPU>;
template struct equal_op<double, device::GPU>;

template struct zeros_op<int, device::GPU>;
template struct zeros_op<float, device::GPU>;
template struct zeros_op<double, device::GPU>;

template struct ones_op<int, device::GPU>;
template struct ones_op<float, device::GPU>;
template struct ones_op<double, device::GPU>;

template struct eye_op<int, device::GPU>;
template struct eye_op<float, device::GPU>;
template struct eye_op<double, device::GPU>;

template struct transpose_op<int, device::GPU>;
template struct transpose_op<float, device::GPU>;
template struct transpose_op<double, device::GPU>;

template struct fc_forward_op<int, device::GPU>;
template struct fc_forward_op<float, device::GPU>;
template struct fc_forward_op<double, device::GPU>;

template struct fc_backward_op<int, device::GPU>;
template struct fc_backward_op<float, device::GPU>;
template struct fc_backward_op<double, device::GPU>;

template struct im2col_op<int, device::GPU>;
template struct im2col_op<float, device::GPU>;
template struct im2col_op<double, device::GPU>;

template struct col2im_op<int, device::GPU>;
template struct col2im_op<float, device::GPU>;
template struct col2im_op<double, device::GPU>;

template struct conv2d_forward_op<int, device::GPU>;
template struct conv2d_forward_op<float, device::GPU>;
template struct conv2d_forward_op<double, device::GPU>;

template struct conv2d_backward_op<int, device::GPU>;
template struct conv2d_backward_op<float, device::GPU>;
template struct conv2d_backward_op<double, device::GPU>;

template struct max_pool_forward_op<int, device::GPU>;
template struct max_pool_forward_op<float, device::GPU>;
template struct max_pool_forward_op<double, device::GPU>;

template struct max_pool_backward_op<int, device::GPU>;
template struct max_pool_backward_op<float, device::GPU>;
template struct max_pool_backward_op<double, device::GPU>;

} // namespace ops
