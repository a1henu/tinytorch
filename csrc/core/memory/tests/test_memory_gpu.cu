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

    using malloc_cpu_op = memory::malloc_mem_op<double, device::CPU>;
    using malloc_gpu_op = memory::malloc_mem_op<double, device::GPU>;

    using free_cpu_op = memory::free_mem_op<double, device::CPU>;
    using free_gpu_op = memory::free_mem_op<double, device::GPU>;

    using copy_c2c_op = memory::copy_mem_op<double, device::CPU, device::CPU>;
    using copy_c2g_op = memory::copy_mem_op<double, device::CPU, device::GPU>;
    using copy_g2c_op = memory::copy_mem_op<double, device::GPU, device::CPU>;
    using copy_g2g_op = memory::copy_mem_op<double, device::GPU, device::GPU>;

    using set_cpu_op = memory::set_mem_op<double, device::CPU>;
    using set_gpu_op = memory::set_mem_op<double, device::GPU>;
};

__global__ void 
assert_arr_eq_val_kernel(bool* result, double* p_data, double val, size_t size) {
    CUDA_KERNEL_LOOP(i, size) {
        if (p_data[i] != val) {
            *result = false;          
        }
    }
}

void assert_arr_eq_val(double* p_data, double val, size_t size) {
    bool* d_result;
    bool h_result = true;
    cudaMalloc(&d_result, sizeof(bool));
    cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
    assert_arr_eq_val_kernel<<<CUDA_GET_BLOCKS(size), CUDA_K_THREADS>>>(d_result, p_data, val, size);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    ASSERT_TRUE(h_result);
}

TEST_F(TestMemory, malloc_CPU) {
    double* p_data = nullptr;
    malloc_cpu_op()(device::cpu_device, p_data, vt_dim);
}

TEST_F(TestMemory, malloc_GPU) {
    double* p_data = nullptr;
    malloc_gpu_op()(device::gpu_device, p_data, vt_dim);
}

TEST_F(TestMemory, free_CPU) {
    double* p_data = nullptr;
    malloc_cpu_op()(device::cpu_device, p_data, vt_dim);
    free_cpu_op()(device::cpu_device, p_data);
}

TEST_F(TestMemory, free_GPU) {
    double* p_data = nullptr;
    malloc_gpu_op()(device::gpu_device, p_data, vt_dim);
    free_gpu_op()(device::gpu_device, p_data);
}

TEST_F(TestMemory, memset_CPU) {
    double* p_data = nullptr;
    malloc_cpu_op()(device::cpu_device, p_data, vt_dim);
    set_cpu_op()(device::cpu_device, p_data, 0, vt_dim);
    for (int i = 0; i < vt_dim; i++) {
        EXPECT_EQ(p_data[i], 0);
    }
    free_cpu_op()(device::cpu_device, p_data);
}

TEST_F(TestMemory, memset_GPU) {
    double* p_data = nullptr;
    malloc_gpu_op()(device::gpu_device, p_data, vt_dim);
    set_gpu_op()(device::gpu_device, p_data, 0, vt_dim);
    assert_arr_eq_val(p_data, 0, vt_dim);
    free_gpu_op()(device::gpu_device, p_data);
}

int main(int argc, char **argv) {
    std::cout << "run test for CORE::MEMORY" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
