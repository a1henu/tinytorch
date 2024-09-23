/**
 * @file test_memory.cpp
 * @brief Memory operator test cases for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include "core/device/device.h"
#include "core/memory/memory.h"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

std::vector<double> generate_random_vector(size_t size, double min_value, double max_value) {
    std::vector<double> vec(size);
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(min_value, max_value); 

    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });

    return vec;
}

class TestMemory : public ::testing::Test
{
protected:
    std::vector<double> v_test;

    int vt_dim;


    void SetUp() override {
        v_test = generate_random_vector(10, 0.0, 1.0); 
        vt_dim = v_test.size();
    }
    void TearDown() override {
    }

    using malloc_op = memory::malloc_mem_op<double, device::CPU>;
    using free_op = memory::free_mem_op<double, device::CPU>;
    using copy_op = memory::copy_mem_op<double, device::CPU, device::CPU>;
    using set_op = memory::set_mem_op<double, device::CPU>;
};

TEST_F(TestMemory, malloc_CPU) {
    double* p_data = nullptr;
    malloc_op()(device::cpu_device, p_data, vt_dim);
}

TEST_F(TestMemory, free_CPU) {
    double* p_data = nullptr;
    malloc_op()(device::cpu_device, p_data, vt_dim);
    free_op()(device::cpu_device, p_data);
}

TEST_F(TestMemory, memcpy_CPU) {
    double* p_data = nullptr;
    malloc_op()(device::cpu_device, p_data, vt_dim);
    copy_op()(device::cpu_device, device::cpu_device, p_data, v_test.data(), vt_dim);
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_EQ(p_data[i], v_test[i]);
    }
    free_op()(device::cpu_device, p_data);
}

TEST_F(TestMemory, memset_CPU) {
    double* p_data = nullptr;
    malloc_op()(device::cpu_device, p_data, vt_dim);
    set_op()(device::cpu_device, p_data, 0, vt_dim);
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_EQ(p_data[i], 0);
    }
    free_op()(device::cpu_device, p_data);
}

int main(int argc, char **argv) {
    std::cout << "run test for CORE::MEMORY" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
