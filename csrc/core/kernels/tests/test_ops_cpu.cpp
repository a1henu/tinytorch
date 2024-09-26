/**
 * @file test_ops_cpu.cpp
 * @brief Math operator test cases for CPU
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
    std::vector<double> vt_1;
    std::vector<double> vt_2;

    int vt_dim;

    void SetUp() override {
        vt_1 = generate_random_vector(100, 0.0, 1.0); 
        vt_2 = generate_random_vector(100, 0.0, 1.0);
        vt_dim = vt_1.size();
    }
    void TearDown() override {
    }
    
    using add_cpu_op = ops::add_op<double, device::CPU>;
    using sub_cpu_op = ops::sub_op<double, device::CPU>;
    using equal_cpu_op = ops::equal_op<double, device::CPU>;

    using add_gpu_op = ops::add_op<double, device::GPU>;
    using sub_gpu_op = ops::sub_op<double, device::GPU>;
    using equal_gpu_op = ops::equal_op<double, device::GPU>;
};

TEST_F(TestOps, TestAddOp_cpu_1) {
    std::vector<double> vt_out(vt_dim);
    add_cpu_op()(device::cpu_device, vt_out.data(), vt_1.data(), vt_2.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out[i], vt_1[i] + vt_2[i]);
    }
}

TEST_F(TestOps, TestAddOp_cpu_2) {
    std::vector<double> vt_out(vt_dim);
    add_cpu_op()(device::cpu_device, vt_out.data(), vt_2.data(), vt_1.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out[i], vt_1[i] + vt_2[i]);
    }
}

TEST_F(TestOps, TestSubOp_cpu) {
    std::vector<double> vt_out(vt_dim);
    sub_cpu_op()(device::cpu_device, vt_out.data(), vt_1.data(), vt_2.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out[i], vt_1[i] - vt_2[i]);
    }
}

TEST_F(TestOps, TestEqualOp_cpu_1) {
    bool vt_out;
    equal_cpu_op()(device::cpu_device, &vt_out, vt_1.data(), vt_2.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        if (vt_1[i] != vt_2[i]) {
            EXPECT_FALSE(vt_out);
            return;
        }
    }
    EXPECT_TRUE(vt_out);
}

TEST_F(TestOps, TestEqualOp_cpu_2) {
    std::vector<double> vt_out_1(vt_dim);
    std::vector<double> vt_out_2(vt_dim);
    add_cpu_op()(device::cpu_device, vt_out_1.data(), vt_1.data(), vt_2.data(), vt_dim);
    add_cpu_op()(device::cpu_device, vt_out_2.data(), vt_2.data(), vt_1.data(), vt_dim);
    bool vt_out;
    equal_cpu_op()(device::cpu_device, &vt_out, vt_out_1.data(), vt_out_2.data(), vt_dim);
    EXPECT_TRUE(vt_out);
}

int main(int argc, char** argv) {
std::cout << "run test for CORE::KERNELS::OPS::CPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
