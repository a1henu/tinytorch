/**
 * @file test_activation_cpu.cpp
 * @brief Activation function test cases for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "tensor/tensor.h"
#include "layers/layers.h"

class TestFC : public ::testing::Test {
protected:
    std::vector<double> x, w, b, y, dy, dx, dw, db;
    tensor::Tensor<double> input, weight_, weight, bias, o, output, input_grad, output_grad, weight_grad, bias_grad;

    int batch_size = 2, in_features = 3, out_features = 4;

    void SetUp() override {
        x = {0.349714, 0.928925, 0.120875, 1.776752, -0.020010, -2.058003};

        w = {2.622056, 0.776671, 0.935029, 0.038836, -0.106238, 
            -1.187628, 0.496029, 0.668119, -0.759459, 1.103246, 
            0.620958, -0.294705};

        b = {1.174965, -0.723839, -1.496529, 1.039356};

        y = {2.167105, 3.066317, -0.699335, 1.567620, -1.227105, 
            1.714296, 1.506132, 3.773981};

        dy =  {-0.299013, -1.662136, 0.646587, -0.417194, 1.058998, 
                -1.708521, -1.830452, 0.362910};

        dx =  {-2.253062, -4.821511, -0.730024, -2.162754, -1.312314, 0.131926};

        dw =  {-1.648568, -2.989346, 3.426663, -0.161421, -0.663093, 
                0.845647, -1.216741, -2.907610, 3.494951, -0.303019, 
                0.423545, -0.710243};

        db = {-1.961148, 0.229394, -0.649523, -1.467542};

        input = tensor::Tensor<double>({batch_size, in_features}, tensor::DeviceType::CPU, x.data());
        weight = tensor::Tensor<double>({in_features, out_features}, tensor::DeviceType::CPU, w.data());
        bias = tensor::Tensor<double>({1, out_features}, tensor::DeviceType::CPU, b.data());
        o = tensor::Tensor<double>({batch_size, out_features}, tensor::DeviceType::CPU);
        output = tensor::Tensor<double>({batch_size, out_features}, tensor::DeviceType::CPU, y.data());
        input_grad = tensor::Tensor<double>({batch_size, in_features}, tensor::DeviceType::CPU);
        output_grad = tensor::Tensor<double>({batch_size, out_features}, tensor::DeviceType::CPU, dy.data());
        weight_grad = tensor::Tensor<double>({in_features, out_features}, tensor::DeviceType::CPU);
        bias_grad = tensor::Tensor<double>({1, out_features}, tensor::DeviceType::CPU);
    }
    void TearDown() override {
    }
};

TEST_F(TestFC, TestFCForward) {
    layers::fc_forward(input, weight, bias, o);

    for (int i = 0; i < batch_size * out_features; ++i) {
        EXPECT_NEAR(o.get_data()[i], y[i], 1e-4);
    }
}

TEST_F(TestFC, TestFCBackward) {
    layers::fc_backward(input, weight, bias, output, input_grad, weight_grad, bias_grad, output_grad);

    for (int i = 0; i < batch_size * in_features; ++i) {
        EXPECT_NEAR(input_grad.get_data()[i], dx[i], 1e-4);
    }

    for (int i = 0; i < in_features * out_features; ++i) {
        EXPECT_NEAR(weight_grad.get_data()[i], dw[i], 1e-4);
    }

    for (int i = 0; i < out_features; ++i) {
        EXPECT_NEAR(bias_grad.get_data()[i], db[i], 1e-4);
    }
}

int main(int argc, char** argv) {
    std::cout << "run test for LAYER::TEST_LAYER" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}