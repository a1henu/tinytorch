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

#ifdef __CUDA
        input.to_gpu();
        weight.to_gpu();
        bias.to_gpu();
        o.to_gpu();
        output.to_gpu();
        input_grad.to_gpu();
        output_grad.to_gpu();
        weight_grad.to_gpu();
        bias_grad.to_gpu();
#endif
    }
    void TearDown() override {
    }
};

class TestPoolingLayer : public ::testing::Test {
protected:
    std::vector<double> x, y, dy, dx;
    tensor::Tensor<double> input, output, output_grad, input_grad;
    std::vector<int> mask;
    tensor::Tensor<int> mask_out, mask_in;

    int batch_size = 1, channels = 2, height = 4, width = 4;
    int kernel_h = 2, kernel_w = 2, pad_h = 0, pad_w = 0, stride_h = 2, stride_w = 2;

    void SetUp() override {
        x = {
            // 1 channels
            -0.5752, 1.1023, 0.8327, -0.3337, 
            -0.0532, 0.8745, 1.4135, -0.4422, 
            -0.4538, 0.2952, 0.4086, -0.3135, 
            0.6764, 0.3422, -0.1896, 0.3065,
            // 2 channels 
            -0.3942, 1.3151, 0.5020, 0.7686, 
            -1.7310, 0.8545, -1.3705, -0.3178, 
            -2.5553, 1.1632, 0.4868, -0.1809,  
            0.0281, 1.2346, 0.3800, 0.2100
        };
        y = {
            1.1023, 1.4135, 
            0.6764, 0.4086, 
            1.3151, 0.7686, 
            1.2346, 0.4868
        };
        dy = {1, 1, 1, 1, 1, 1, 1, 1};
        dx = {
            0, 1, 0, 0, 
            0, 0, 1, 0, 
            0, 0, 1, 0, 
            1, 0, 0, 0, 
            0, 1, 0, 1, 
            0, 0, 0, 0, 
            0, 0, 1, 0, 
            0, 1, 0, 0
        };
        mask = {1, 2, 2, 0, 1, 1, 3, 0};

        input = tensor::Tensor<double>({batch_size, channels, height, width}, tensor::DeviceType::CPU, x.data());
        output = tensor::Tensor<double>({batch_size, channels, height/2, width/2}, tensor::DeviceType::CPU);
        output_grad = tensor::Tensor<double>({batch_size, channels, height/2, width/2}, tensor::DeviceType::CPU, dy.data());
        input_grad = tensor::Tensor<double>({batch_size, channels, height, width}, tensor::DeviceType::CPU);
        mask_out = tensor::Tensor<int>({batch_size, channels, height/2, width/2}, tensor::DeviceType::CPU);

        mask_in = tensor::Tensor<int>({batch_size, channels, height/2, width/2}, tensor::DeviceType::CPU, mask.data());
#ifdef __CUDA
        input.to_gpu();
        output.to_gpu();
        output_grad.to_gpu();
        input_grad.to_gpu();
        mask_out.to_gpu();
        mask_in.to_gpu();
#endif
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

#ifdef __CUDA
        input.to_gpu();
        output.to_gpu();
#endif
    }
    void TearDown() override {
    }
};

class TestCrossEntropyLayer : public ::testing::Test {
protected:
    std::vector<double> z_o, z, g;
    std::vector<int> t;

    tensor::Tensor<double> input, input_softmax, grad;
    tensor::Tensor<double> output;
    tensor::Tensor<int> target;

    double expected_loss = 2.6912;

    int batch_size = 5, num_classes = 10;

    void SetUp() override {
        z_o = {
            -1.028684, 0.856440, 1.369762, -1.437391, 1.551560, 1.139737, -1.240337, -0.648702, -0.400014, 1.586942, 
            2.365771, 2.535360, -0.772002, 0.039393, -1.142135, 1.507503, 0.550930, 0.630071, -0.746441, 0.497415, 
            -0.382562, -1.579024, 1.228670, -0.061057, -0.585326, -1.225693, -0.035275, 0.099546, 0.465645, 0.714231, 
            -0.739603, 0.209539, 0.564118, 0.357420, -0.649761, 1.078385, -0.351789, -1.801129, -0.612122, -0.219620, 
            0.764168, -1.062313, 0.094680, -0.484254, -1.003578, 0.560764, -0.030785, 0.453219, 0.187955, 0.185473
        };
        z = {
            0.016942, 0.111600, 0.186465, 0.011258, 0.223640, 0.148149, 0.013710, 0.024774, 0.031768, 0.231695, 
            0.301412, 0.357118, 0.013075, 0.029432, 0.009030, 0.127767, 0.049089, 0.053132, 0.013414, 0.046531, 
            0.057797, 0.017470, 0.289503, 0.079714, 0.047189, 0.024874, 0.081795, 0.093601, 0.134982, 0.173075, 
            0.045141, 0.116621, 0.166253, 0.135208, 0.049384, 0.278044, 0.066527, 0.015616, 0.051279, 0.075927, 
            0.190346, 0.030642, 0.097452, 0.054621, 0.032495, 0.155313, 0.085961, 0.139477, 0.106979, 0.106714
        };
        g = {
            0.0034, 0.0223, 0.0373, -0.1977, 0.0447, 0.0296, 0.0027, 0.0050, 0.0064, 0.0463,
            0.0603, 0.0714, 0.0026, 0.0059, 0.0018, 0.0256, 0.0098, -0.1894, 0.0027, 0.0093,
            0.0116, 0.0035, 0.0579, 0.0159, 0.0094, 0.0050, 0.0164, 0.0187, -0.1730, 0.0346,
            0.0090, 0.0233, -0.1667, 0.0270, 0.0099, 0.0556, 0.0133, 0.0031, 0.0103, 0.0152,
            0.0381, 0.0061, 0.0195, 0.0109, 0.0065, 0.0311, 0.0172, 0.0279, 0.0214, -0.1787
        };
        t = {
            3, 7, 8, 2, 9
        };

        input = tensor::Tensor<double>({batch_size, num_classes}, tensor::DeviceType::CPU, z_o.data());
        input_softmax = tensor::Tensor<double>({batch_size, num_classes}, tensor::DeviceType::CPU, z.data());
        grad = tensor::Tensor<double>({batch_size, num_classes}, tensor::DeviceType::CPU);
        target = tensor::Tensor<int>({batch_size}, tensor::DeviceType::CPU, t.data());
        output = tensor::Tensor<double>({1}, tensor::DeviceType::CPU);

#ifdef __CUDA
        input.to_gpu();
        input_softmax.to_gpu();
        grad.to_gpu();
        target.to_gpu();
        output.to_gpu();
#endif
    }
    void TearDown() override {
    }
};

TEST_F(TestFCLayer, TestFCForward) {
    layers::fc_forward(input, weight, bias, o);

#ifdef __CUDA
    o.to_cpu();
#endif

    for (int i = 0; i < batch_size * out_features; ++i) {
        EXPECT_NEAR(o.get_data()[i], y[i], 1e-4);
    }
}

TEST_F(TestFCLayer, TestFCBackward) {
    layers::fc_backward(input, weight, bias, output, input_grad, weight_grad, bias_grad, output_grad);

#ifdef __CUDA
    input_grad.to_cpu();
    weight_grad.to_cpu();
    bias_grad.to_cpu();
#endif

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

#ifdef __CUDA
    output.to_cpu();
#endif

    for (int i = 0; i < batch_size * num_classes; ++i) {
        EXPECT_NEAR(output.get_data()[i], y[i], 1e-4);
    }
}

TEST_F(TestCrossEntropyLayer, TestCrossEntropyForward) {
    layers::cross_entropy_forward(input_softmax, target, output);

#ifdef __CUDA
    output.to_cpu();
#endif

    EXPECT_NEAR(*output.get_data(), expected_loss, 1e-4);
}

TEST_F(TestCrossEntropyLayer, TestCrossEntropyBackward) {
    layers::cross_entropy_backward(input, target, grad);

#ifdef __CUDA
    grad.to_cpu();
#endif

    for (int i = 0; i < batch_size * num_classes; ++i) {
        EXPECT_NEAR(grad.get_data()[i], g[i], 1e-4);
    }
}

TEST_F(TestPoolingLayer, TestPoolingForward) {
    layers::max_pool_forward(
        input,
        mask_out,
        output,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w
    );

#ifdef __CUDA
    output.to_cpu();
    mask_out.to_cpu();
#endif

    for (int i = 0; i < batch_size * channels * height/2 * width/2; ++i) {
        EXPECT_NEAR(output.get_data()[i], y[i], 1e-4);
        EXPECT_EQ(mask_out.get_data()[i], mask[i]);
    }
}

TEST_F(TestPoolingLayer, TestPoolingBackward) {
    layers::max_pool_backward(
        input_grad,
        mask_in,
        output_grad,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w
    );

#ifdef __CUDA
    input_grad.to_cpu();
#endif

    for (int i = 0; i < batch_size * channels * height * width; ++i) {
        std::cout << input_grad.get_data()[i] << " " << dx[i] << std::endl;
        EXPECT_NEAR(input_grad.get_data()[i], dx[i], 1e-4);
    }
}

int main(int argc, char** argv) {
    std::cout << "run test for LAYER::TEST_LAYER" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}