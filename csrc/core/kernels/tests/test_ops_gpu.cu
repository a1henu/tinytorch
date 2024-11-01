/**
 * @file test_ops_gpu.cu
 * @brief Math operator test cases for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

#include "core/device/device.h"
#include "core/kernels/ops.h"

#include "error/error.h"

std::vector<float> generate_random_vector(size_t size, float min_value, float max_value) {
    std::vector<float> vec(size);
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(min_value, max_value); 

    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });

    return vec;
}

std::vector<double> generate_random_vector(size_t size, double min_value, double max_value) {
    std::vector<double> vec(size);
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(min_value, max_value); 

    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });

    return vec;
}

class TestOps : public ::testing::Test {
protected:
    std::vector<double> v1;
    std::vector<double> v2;

    double* vt_1;
    double* vt_2;

    int vt_dim;

    void SetUp() override {
        v1 = generate_random_vector(100, 0.0, 1.0); 
        v2 = generate_random_vector(100, 0.0, 1.0);
        vt_dim = v1.size();

        cudaMalloc(&vt_1, vt_dim * sizeof(double));
        cudaMalloc(&vt_2, vt_dim * sizeof(double));

        cudaMemcpy(vt_1, v1.data(), vt_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(vt_2, v2.data(), vt_dim * sizeof(double), cudaMemcpyHostToDevice);
    }
    void TearDown() override {
        cudaFree(vt_1);
        cudaFree(vt_2);
    }
    
    using add_cpu_op = ops::add_op<double, device::CPU>;
    using sub_cpu_op = ops::sub_op<double, device::CPU>;
    using smatmul_cpu_op = ops::matmul_op<float, device::CPU>;
    using dmatmul_cpu_op = ops::matmul_op<double, device::CPU>;
    using equal_cpu_op = ops::equal_op<double, device::CPU>;
    using ones_cpu_op = ops::ones_op<double, device::CPU>;
    using eye_cpu_op = ops::eye_op<double, device::CPU>;
    using im2col_cpu_op = ops::im2col_op<int, device::CPU>;

    using add_gpu_op = ops::add_op<double, device::GPU>;
    using sub_gpu_op = ops::sub_op<double, device::GPU>;
    using smatmul_gpu_op = ops::matmul_op<float, device::GPU>;
    using dmatmul_gpu_op = ops::matmul_op<double, device::GPU>;
    using equal_gpu_op = ops::equal_op<double, device::GPU>;
    using ones_gpu_op = ops::ones_op<double, device::GPU>;
    using eye_gpu_op = ops::eye_op<double, device::GPU>;
    using im2col_gpu_op = ops::im2col_op<int, device::GPU>;
};

TEST_F(TestOps, TestAddOp_gpu_1) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    add_gpu_op()(device::gpu_device, vt_out, vt_1, vt_2, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], v1[i] + v2[i]);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestAddOp_gpu_2) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    add_gpu_op()(device::gpu_device, vt_out, vt_2, vt_1, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], v1[i] + v2[i]);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestSubOp_gpu) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    sub_gpu_op()(device::gpu_device, vt_out, vt_1, vt_2, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], v1[i] - v2[i]);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestMatmulOp_gpu_float) {
    const int m = 30, n = 40, k = 35;
    std::vector<float> A = generate_random_vector(m * k, 0.0f, 1.0f);
    std::vector<float> B = generate_random_vector(k * n, 0.0f, 1.0f);
    float* d_A;
    float* d_B;
    float* d_C;
    float* h_C = new float[m * n];

    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    smatmul_gpu_op()(device::gpu_device, "N", "N", m, n, k, alpha, d_A, k, d_B, n, beta, d_C, n);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> C_expected(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_C[i], C_expected[i], 1e-4f);
    }

    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(TestOps, TestMatmulOp_gpu_double) {
    const int m = 2, n = 3, k = 4;
    std::vector<double> A = generate_random_vector(m * k, 0.0, 1.0);
    std::vector<double> B = generate_random_vector(k * n, 0.0, 1.0);
    double* d_A;
    double* d_B;
    double* d_C;
    double* h_C = new double[m * n];

    cudaMalloc(&d_A, m * k * sizeof(double));
    cudaMalloc(&d_B, k * n * sizeof(double));
    cudaMalloc(&d_C, m * n * sizeof(double));

    cudaMemcpy(d_A, A.data(), m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(double), cudaMemcpyHostToDevice);

    const double alpha = 1.0;
    const double beta = 0.0;

    dmatmul_gpu_op()(device::gpu_device, "N", "N", m, n, k, alpha, d_A, k, d_B, n, beta, d_C, n);

    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> C_expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_C[i], C_expected[i], 1e-4);
    }

    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(TestOps, TestEqualOp_gpu_1) {
    bool vt_out_c = true;
    bool* vt_out_g;
    cudaMalloc(&vt_out_g, sizeof(bool));
    equal_gpu_op()(device::gpu_device, vt_out_g, vt_1, vt_2, vt_dim);
    cudaMemcpy(&vt_out_c, vt_out_g, sizeof(bool), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        if (v1[i] != v2[i]) {
            EXPECT_FALSE(vt_out_c);
            cudaFree(vt_out_g);
            return;
        }
    }
    EXPECT_TRUE(vt_out_c);
    cudaFree(vt_out_g);
}

TEST_F(TestOps, TestEqualOp_gpu_2) {
    double* vt_out1;
    double* vt_out2;
    bool* vt_out_g;
    bool* vt_out_c = new bool;
    cudaMalloc(&vt_out1, vt_dim * sizeof(double));
    cudaMalloc(&vt_out2, vt_dim * sizeof(double));
    cudaMalloc(&vt_out_g, sizeof(bool));
    add_gpu_op()(device::gpu_device, vt_out1, vt_1, vt_2, vt_dim);
    add_gpu_op()(device::gpu_device, vt_out2, vt_2, vt_1, vt_dim);
    equal_gpu_op()(device::gpu_device, vt_out_g, vt_out1, vt_out2, vt_dim);
    cudaMemcpy(vt_out_c, vt_out_g, sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(*vt_out_c);
    cudaFree(vt_out1);
    cudaFree(vt_out2);
    cudaFree(vt_out_g);
    delete vt_out_c;
}

TEST_F(TestOps, TestOnesOp_gpu) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    ones_gpu_op()(device::gpu_device, vt_out, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], 1.0);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestEyeOp_gpu) {
    const int dim = 100;
    double* vt_out;
    double* vt_out_cpu = new double[dim * dim];
    cudaMalloc(&vt_out, dim * dim * sizeof(double));
    eye_gpu_op()(device::gpu_device, vt_out, dim);
    cudaMemcpy(vt_out_cpu, vt_out, dim * dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i == j) {
                EXPECT_EQ(vt_out_cpu[i * dim + j], 1.0);
            } else {
                EXPECT_EQ(vt_out_cpu[i * dim + j], 0.0);
            }
        }
    }
}

TEST_F(TestOps, TestIm2ColOp_gpu) {
    /**
     * example img is 
     * [
     * [1, 2, 3;
     *  4, 5, 6;
     *  7, 8, 9],
     * [3, 2, 1;
     *  6, 5, 4;
     *  9, 8, 7]
     * ]
     * 
     * so data_im is
     * [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7]
     * 
     * im2col(img)(2X2 kernel, 1X1 stride, 0 padding) is
     * [
     *  1, 2, 4, 5, 3, 2, 6, 5;
     *  2, 3, 5, 6, 2, 1, 5, 4;
     *  4, 5, 7, 8, 6, 5, 9, 8;
     *  5, 6, 8, 9, 5, 4, 8, 7
     * ]
     * 
     * so data_col is
     * [1, 2, 4, 5, 3, 2, 6, 5, 2, 3, 5, 6, 2, 1, 5, 4, 4, 5, 7, 8, 6, 5, 9, 8, 5, 6, 8, 9, 5, 4, 8, 7]
     */
    int data_im[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7};
    int data_col[32] = {0};
    int gt_col[32] = {1, 2, 4, 5, 3, 2, 6, 5, 2, 3, 5, 6, 2, 1, 5, 4, 4, 5, 7, 8, 6, 5, 9, 8, 5, 6, 8, 9, 5, 4, 8, 7};

    int* gdata_im;
    int* gdata_col;
    cudaMalloc(&gdata_im, 18 * sizeof(int));
    cudaMalloc(&gdata_col, 32 * sizeof(int));

    cudaMemcpy(gdata_im, data_im, 18 * sizeof(int), cudaMemcpyHostToDevice);
    im2col_gpu_op()(device::gpu_device, gdata_im, gdata_col, 2, 3, 3, 2, 2, 0, 0, 1, 1);
    cudaMemcpy(data_col, gdata_col, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(data_col[i], gt_col[i]);
    }
}

int main(int argc, char** argv) {
std::cout << "run test for CORE::KERNELS::OPS::GPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
