/**
 * @file test_tensor.cpp
 * @brief Tensor class test cases
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "tensor/tensor.h"
#include "core/device/device.h"

class TestTensor : public ::testing::Test {
protected:
    std::vector<int> shape;
    std::vector<double> v;

    void SetUp() override {
        shape = {2, 3, 4};
        v = {
            0.760609, -0.715157, 0.048647, -0.090885, 0.849236, 
            1.422869, 1.486288, 1.030767, 0.924290, -1.496499
        };
    }

    void TearDown() override {
    }
};

TEST_F(TestTensor, default_constructor) {
    tensor::Tensor<double> t;
}

TEST_F(TestTensor, shape_device_constructor) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU);
}

TEST_F(TestTensor, shape_data_t_device_d_device_constructor) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU, v.data());
    for (int i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], t.get_data()[i]);
    }
}

TEST_F(TestTensor, tensor_cpu_to_cpu) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU);
    tensor::Tensor<double> t_cpu = t.cpu();

    ASSERT_EQ(t_cpu.in_cpu(), true);
}

#ifdef __CUDA

TEST_F(TestTensor, tensor_gpu_to_cpu) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::GPU);
    tensor::Tensor<double> t_cpu = t.cpu();
    
    ASSERT_EQ(t_cpu.in_cpu(), true);
}

TEST_F(TestTensor, tensor_cpu_to_gpu) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU);
    tensor::Tensor<double> t_gpu = t.gpu();
    
    ASSERT_EQ(t_gpu.in_gpu(), true);
}

TEST_F(TestTensor, tensor_gpu_to_gpu) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::GPU);
    tensor::Tensor<double> t_gpu = t.gpu();
    
    ASSERT_EQ(t_gpu.in_gpu(), true);
}

TEST_F(TestTensor, tensor_to_cpu) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::GPU);
    t.to_cpu();
    
    ASSERT_EQ(t.in_cpu(), true);
}

TEST_F(TestTensor, tensor_to_gpu) {
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU);
    t.to_gpu();
    
    ASSERT_EQ(t.in_gpu(), true);
}

#endif

TEST_F(TestTensor, assignment_operator) {
    tensor::Tensor<double> t1;
    t1 = tensor::Tensor<double>(shape, tensor::DeviceType::CPU);
}

TEST_F(TestTensor, copy_constructor) {
    tensor::Tensor<double> t1(shape, tensor::DeviceType::CPU);
    tensor::Tensor<double> t2(t1);
}

TEST_F(TestTensor, reshape_without_autocalc) {
#ifndef __CUDA
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU);
    tensor::Tensor<double> t_reshape = t.reshape({3, 2, 4});
#else
    tensor::Tensor<double> tg(shape, tensor::DeviceType::GPU);
    tensor::Tensor<double> tg_reshape = tg.reshape({3, 2, 4});
    tensor::Tensor<double> t_reshape = tg_reshape.cpu();
#endif
    ASSERT_EQ(t_reshape.get_shape()[0], 3);
    ASSERT_EQ(t_reshape.get_shape()[1], 2);
    ASSERT_EQ(t_reshape.get_shape()[2], 4);
}

TEST_F(TestTensor, reshape_with_autocalc) {
#ifndef __CUDA
    tensor::Tensor<double> t(shape, tensor::DeviceType::CPU);
    tensor::Tensor<double> t_reshape = t.reshape({3, 1, -1});
#else
    tensor::Tensor<double> tg(shape, tensor::DeviceType::GPU);
    tensor::Tensor<double> tg_reshape = tg.reshape({3, 1, -1});
    tensor::Tensor<double> t_reshape = tg_reshape.cpu();
#endif
    ASSERT_EQ(t_reshape.get_shape()[0], 3);
    ASSERT_EQ(t_reshape.get_shape()[1], 1);
    ASSERT_EQ(t_reshape.get_shape()[2], 8);
}

TEST_F(TestTensor, transpose) {
#ifndef __CUDA
    tensor::Tensor<double> t({2, 5}, tensor::DeviceType::CPU, v.data());
    tensor::Tensor<double> t_transpose = t.transpose();
#else
    tensor::Tensor<double> tg({2, 5}, tensor::DeviceType::GPU, v.data());
    tensor::Tensor<double> tg_transpose = tg.transpose();
    tensor::Tensor<double> t = tg.cpu();
    tensor::Tensor<double> t_transpose = tg_transpose.cpu();
#endif
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 5; ++j) {
            double a = t[{i, j}];
            double b = t_transpose[{j, i}];
            ASSERT_EQ(a, b);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "run test for TENSOR" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}