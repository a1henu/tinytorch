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

TEST_F(TestTensor, assignment_operator) {
    tensor::Tensor<double> t1;
    t1 = tensor::Tensor<double>(shape, tensor::DeviceType::CPU);
}

TEST_F(TestTensor, copy_constructor) {
    tensor::Tensor<double> t1(shape, tensor::DeviceType::CPU);
    tensor::Tensor<double> t2(t1);
}



int main(int argc, char** argv) {
    std::cout << "run test for TENSOR" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}