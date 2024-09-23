/**
 * @file test_memory_gpu.cu
 * @brief Memory operator test cases for CPU
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
#include "core/memory/memory.h"
#include "macros.h"

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

    using malloc_op = memory::malloc_mem_op<double, device::GPU>;
    using free_op = memory::free_mem_op<double, device::GPU>;
    using copy_c2g_op = memory::copy_mem_op<double, device::CPU, device::GPU>;
    using copy_g2c_op = memory::copy_mem_op<double, device::GPU, device::CPU>;
    using copy_g2g_op = memory::copy_mem_op<double, device::GPU, device::GPU>;
    using set_op = memory::set_mem_op<double, device::GPU>;
};

__global__ void 
assert_arr_eq_val_kernel(double* p_data, double val, size_t dim) {
    CUDA_KERNEL_LOOP(idx, dim) {
        
    }
}

TEST_F(TestMemory, malloc_GPU) {
    double* p_data = nullptr;
    malloc_op()(device::gpu_device, p_data, vt_dim);
}

TEST_F(TestMemory, free_GPU) {
    double* p_data = nullptr;
    malloc_op()(device::gpu_device, p_data, vt_dim);
    free_op()(device::gpu_device, p_data);
}

TEST_F(TestMemory, set_op) {
    double* p_data = nullptr;
    malloc_op()(device::gpu_device, p_data, vt_dim);
    set_op()(device::gpu_device, p_data, 0, vt_dim);
    assert_arr_eq_val_kernel<<<CUDA_GET_BLOCKS(vt_dim), CUDA_K_THREADS>>>(p_data, 0, vt_dim);
    free_op()(device::gpu_device, p_data);
}

int main(int argc, char **argv) {
    std::cout << "run test for CORE::MEMORY" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
