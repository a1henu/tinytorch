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
    using equal_cpu_op = ops::equal_op<double, device::CPU>;

    using add_gpu_op = ops::add_op<double, device::GPU>;
    using sub_gpu_op = ops::sub_op<double, device::GPU>;
    using equal_gpu_op = ops::equal_op<double, device::GPU>;
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

int main(int argc, char** argv) {
std::cout << "run test for CORE::KERNELS::OPS::GPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
