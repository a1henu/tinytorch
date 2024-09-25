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

    void SetUp() override {
        shape = {2, 3, 4};
    }

    void TearDown() override {
    }
};

TEST_F(TestTensor, default_constructor) {
    tensor::Tensor<double> t;
}

TEST_F(TestTensor, shape_device_constructor) {
    tensor::Tensor<double> t(shape, new device::CPU());
}

TEST_F(TestTensor, shape_data_t_device_d_device_constructor) {
    double* data = new double[24];
    tensor::Tensor<double> t(shape, new device::CPU(), data);
}

int main(int argc, char** argv) {
    std::cout << "run test for TENSOR" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}