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

} // namespace ops