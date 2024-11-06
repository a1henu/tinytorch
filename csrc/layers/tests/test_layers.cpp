/**
 * @file test_layers_cpu.cpp
 * @brief Fully connected layer test cases for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "tensor/tensor.h"
#include "layers/layers.h"

class TestFCLayer : public ::testing::Test {
protected:
    std::vector<double> x, w, b, y, dy, dx, dw, db;
    tensor::Tensor<double> input, weight_, weight, bias, o, output, input_grad, output_grad, weight_grad, bias_grad;

    int batch_size = 2, in_features = 3, out_features = 4;

    void SetUp() override {
        x = {-0.148011, -0.729241, 2.118632, -1.679356, -0.540102, 0.861272};

        w = {-0.751523, -1.105763, 0.287860, 0.683705, -0.067248, -0.521211, 0.934177, -0.972594, 0.041104, -0.229331, -1.562299, -1.330761};

        b = {-1.118533, -0.789384, 1.793169, -1.068506};

        y = {-0.871176, -0.731499, -2.240614, -3.279839, 0.215264, 1.151577, -0.540365, -2.837536};

        dy = {-0.447217, 0.134503, -0.906900, -0.529319, 1.357754, -0.369581, 0.877381, -1.313340};

        dx = {-0.435592, -0.372422, 2.072017, -1.257088, 2.198299, 0.517575};

        dw = {-2.213959, 0.600751, -1.339203, 2.283910, -0.407197, 0.101527, 0.187473, 1.095338, 0.221908, -0.033348, -1.165723, -2.252574};

        db = {0.910537, -0.235078, -0.029519, -1.842658};

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

class TestSoftmaxLayer : public ::testing::Test {
protected:
    std::vector<double> x, y;
    tensor::Tensor<double> input, output;

    int batch_size = 5, num_classes = 10;

    void SetUp() override {
        x = {
            -1.028684, 0.856440, 1.369762, -1.437391, 1.551560, 1.139737, -1.240337, -0.648702, -0.400014, 1.586942, 
            2.365771, 2.535360, -0.772002, 0.039393, -1.142135, 1.507503, 0.550930, 0.630071, -0.746441, 0.497415, 
            -0.382562, -1.579024, 1.228670, -0.061057, -0.585326, -1.225693, -0.035275, 0.099546, 0.465645, 0.714231, 
            -0.739603, 0.209539, 0.564118, 0.357420, -0.649761, 1.078385, -0.351789, -1.801129, -0.612122, -0.219620, 
            0.764168, -1.062313, 0.094680, -0.484254, -1.003578, 0.560764, -0.030785, 0.453219, 0.187955, 0.185473
        };

        y = {
            0.016942, 0.111600, 0.186465, 0.011258, 0.223640, 0.148149, 0.013710, 0.024774, 0.031768, 0.231695, 
            0.301412, 0.357118, 0.013075, 0.029432, 0.009030, 0.127767, 0.049089, 0.053132, 0.013414, 0.046531, 
            0.057797, 0.017470, 0.289503, 0.079714, 0.047189, 0.024874, 0.081795, 0.093601, 0.134982, 0.173075, 
            0.045141, 0.116621, 0.166253, 0.135208, 0.049384, 0.278044, 0.066527, 0.015616, 0.051279, 0.075927, 
            0.190346, 0.030642, 0.097452, 0.054621, 0.032495, 0.155313, 0.085961, 0.139477, 0.106979, 0.106714
        };

        input = tensor::Tensor<double>({batch_size, num_classes}, tensor::DeviceType::CPU, x.data());
        output = tensor::Tensor<double>({batch_size, num_classes}, tensor::DeviceType::CPU);
    }
    void TearDown() override {
    }
};

TEST_F(TestFCLayer, TestFCForward) {
    layers::fc_forward(input, weight, bias, o);

    for (int i = 0; i < batch_size * out_features; ++i) {
        EXPECT_NEAR(o.get_data()[i], y[i], 1e-4);
    }
}

TEST_F(TestFCLayer, TestFCBackward) {
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

TEST_F(TestSoftmaxLayer, TestSoftmaxForward) {
    layers::softmax_forward(input, output);

    for (int i = 0; i < batch_size * num_classes; ++i) {
        EXPECT_NEAR(output.get_data()[i], y[i], 1e-4);
    }
}

int main(int argc, char** argv) {
    std::cout << "run test for LAYER::TEST_LAYER" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}