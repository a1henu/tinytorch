/**
 * @file test_tensor_ops.cpp
 * @brief Tensor math operator test cases
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

#include "tensor/tensor.h"

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
    std::vector<double> vt_1;
    std::vector<double> vt_2;

    tensor::Tensor<double> tt_1;
    tensor::Tensor<double> tt_2;

    int vt_dim;

#ifdef __CUDA
    double* tt1;
    double* tt2;
#endif

    void SetUp() override {
        vt_1 = generate_random_vector(100, 0.0, 1.0); 
        vt_2 = generate_random_vector(100, 0.0, 1.0);
        vt_dim = vt_1.size();
#ifndef __CUDA
        tt_1 = tensor::Tensor<double>({vt_dim}, tensor::DeviceType::CPU, vt_1.data());
        tt_2 = tensor::Tensor<double>({vt_dim}, tensor::DeviceType::CPU, vt_2.data());
#else
        memory::malloc_mem_op<double, device::GPU>()(
            device::gpu_device, tt1, vt_dim);
        memory::malloc_mem_op<double, device::GPU>()(
            device::gpu_device, tt2, vt_dim);
        memory::copy_mem_op<double, device::GPU, device::CPU>()(
            device::gpu_device, device::cpu_device, tt1, vt_1.data(), vt_dim);
        memory::copy_mem_op<double, device::GPU, device::CPU>()(
            device::gpu_device, device::cpu_device, tt2, vt_2.data(), vt_dim);
        tt_1 = tensor::Tensor<double>({vt_dim}, tensor::DeviceType::GPU, tt1);
        tt_2 = tensor::Tensor<double>({vt_dim}, tensor::DeviceType::GPU, tt2);
#endif
    }
    void TearDown() override {
#ifdef __CUDA
        memory::free_mem_op<double, device::GPU>()(device::gpu_device, tt1);
        memory::free_mem_op<double, device::GPU>()(device::gpu_device, tt2);
#endif
    }
};

#ifndef __CUDA

TEST_F(TestOps, tensor_add_cpu) {
    tensor::Tensor<double> vt_add = tt_1 + tt_2;
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_NEAR(vt_add.get_data()[i], vt_1[i] + vt_2[i], 1e-6);
    }
}

TEST_F(TestOps, tensor_sub_cpu) {
    tensor::Tensor<double> vt_sub = tt_1 - tt_2;
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_NEAR(vt_sub.get_data()[i], vt_1[i] - vt_2[i], 1e-6);
    }
}

TEST_F(TestOps, tensor_mul_cpu) {
    tensor::Tensor<double> tt_1_reshape = tt_1.reshape({5, 20});
    tensor::Tensor<double> tt_2_reshape = tt_2.reshape({20, 5});
    tensor::Tensor<double> vt_mul = tt_1_reshape * tt_2_reshape;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            double sum = 0.0;
            for (int k = 0; k < 20; k++) {
                sum += vt_1[i * 20 + k] * vt_2[k * 5 + j];
            }
            EXPECT_NEAR(vt_mul[{i, j}], sum, 1e-6);
        }
    }
}

TEST_F(TestOps, tensor_eq_cpu) {
    tensor::Tensor<double> vt_1 = tt_1 + tt_2;
    tensor::Tensor<double> vt_2 = tt_2 + tt_1;
    EXPECT_TRUE(vt_1 == vt_1);
}

#else

TEST_F(TestOps, tensor_add_gpu) {
    tensor::Tensor<double> vt_add = tt_1 + tt_2;
    tensor::Tensor<double> vt_ac = vt_add.cpu();
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_NEAR(vt_ac.get_data()[i], vt_1[i] + vt_2[i], 1e-6);
    }
}

TEST_F(TestOps, tensor_sub_gpu) {
    tensor::Tensor<double> vt_sub = tt_1 - tt_2;
    tensor::Tensor<double> vt_sc = vt_sub.cpu();
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_NEAR(vt_sc.get_data()[i], vt_1[i] - vt_2[i], 1e-6);
    }
}

TEST_F(TestOps, tensor_eq_gpu) {
    tensor::Tensor<double> vt_1 = tt_1 + tt_2;
    tensor::Tensor<double> vt_2 = tt_2 + tt_1;
    EXPECT_TRUE(vt_1 == vt_1);
}

#endif

int main(int argc, char** argv) {
    std::cout << "run test for TENSOR::MATH_OP" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}