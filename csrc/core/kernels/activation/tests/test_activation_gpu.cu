/**
 * @file test_activation_gpu.cu
 * @brief Activation function test cases for GPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "core/device/device.h"
#include "core/kernels/activation/relu.h"
#include "core/kernels/activation/sigmoid.h"
#include "core/kernels/activation/softmax.h"

class TestReLU : public ::testing::Test {
protected:
    std::vector<double> v;
    std::vector<double> g;

    std::vector<double> v_relu_f;
    std::vector<double> v_relu_b;

    double* v_g;
    double* g_g;
    double* v_relu_f_g;
    double* v_relu_b_g;

    int v_dim;

    void SetUp() override {
        v = {
            0.021779, -0.091378, 2.529141, -1.314787, 2.063127, 
            0.499841, 0.930395, 1.301085, 3.620728, -0.509598, 
            -0.729976, 1.701381, -0.519704, 0.361764, -0.010430, 
            0.764627, 0.749973, 0.889580, 0.072533, 0.252502, 
            0.179101, 2.111640, 0.788848, -0.130065, 1.355981, 
            0.541689, 0.206137, 1.232952, 0.943046, -0.229882
        };
        g = {
            0.611874, 0.043436, 0.710300, -0.144175, 0.307186, 
            -0.469658, 0.082649, -0.185967, -1.892874, -1.613372, 
            1.128987, -1.476294, 0.511537, 2.049930, 0.040707, 
            -0.332097, 0.460975, 0.286529, -0.167816, -1.922494, 
            0.093031, -0.290727, 0.196690, -0.644933, -0.173954, 
            -0.749864, 0.717477, -0.122634, 0.127579, 2.801707
        };
        v_relu_f = {
            0.021779, 0.000000, 2.529141, 0.000000, 2.063127, 
            0.499841, 0.930395, 1.301085, 3.620728, 0.000000, 
            0.000000, 1.701381, 0.000000, 0.361764, 0.000000, 
            0.764627, 0.749973, 0.889580, 0.072533, 0.252502, 
            0.179101, 2.111640, 0.788848, 0.000000, 1.355981, 
            0.541689, 0.206137, 1.232952, 0.943046, 0.000000
        };
        v_relu_b = {
            0.611874, 0.000000, 0.710300, 0.000000, 0.307186, 
            -0.469658, 0.082649, -0.185967, -1.892874, 0.000000, 
            0.000000, -1.476294, 0.000000, 2.049930, 0.000000, 
            -0.332097, 0.460975, 0.286529, -0.167816, -1.922494, 
            0.093031, -0.290727, 0.196690, 0.000000, -0.173954, 
            -0.749864, 0.717477, -0.122634, 0.127579, 0.000000
        };
        v_dim = v.size();

        cudaMalloc(&v_g, v_dim * sizeof(double));
        cudaMalloc(&g_g, v_dim * sizeof(double));
        cudaMalloc(&v_relu_f_g, v_dim * sizeof(double));
        cudaMalloc(&v_relu_b_g, v_dim * sizeof(double));

        cudaMemcpy(v_g, v.data(), v_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(g_g, g.data(), v_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(v_relu_f_g, v_relu_f.data(), v_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(v_relu_b_g, v_relu_b.data(), v_dim * sizeof(double), cudaMemcpyHostToDevice);
    }
    void TearDown() override {
        cudaFree(v_g);
        cudaFree(g_g);
        cudaFree(v_relu_f_g);
        cudaFree(v_relu_b_g);
    }

};

class TestSigmoid : public ::testing::Test {
protected:
    std::vector<double> x;
    std::vector<double> g;

    std::vector<double> x_sigmoid_f;
    std::vector<double> x_sigmoid_b;

    double* x_g;
    double* g_g;
    double* x_sigmoid_f_g;
    double* x_sigmoid_b_g;

    int x_dim;

    void SetUp() override {
        x = {
            0.760609, -0.715157, 0.048647, -0.090885, 0.849236, 
            1.422869, 1.486288, 1.030767, 0.924290, -1.496499, 
            1.142868, 0.366450, 0.224117, 2.006245, 0.116255, 
            0.295170, -1.305522, 0.590613, -0.296552, -0.988433, 
            -1.320542, 0.617037, 0.462768, -1.011641, -0.022526, 
            -0.207176, 1.544988, -0.053472, -0.179526, -1.048691
        };
        g = {
            0.611874, 0.043436, 0.710300, -0.144175, 0.307186, 
            -0.469658, 0.082649, -0.185967, -1.892874, -1.613372, 
            1.128987, -1.476294, 0.511537, 2.049930, 0.040707, 
            -0.332097, 0.460975, 0.286529, -0.167816, -1.922494, 
            0.093031, -0.290727, 0.196690, -0.644933, -0.173954, 
            -0.749864, 0.717477, -0.122634, 0.127579, 2.801707
        };
        x_sigmoid_f = {
            0.681486, 0.328460, 0.512159, 0.477294, 0.700407, 
            0.805788, 0.815521, 0.737065, 0.715915, 0.182948, 
            0.758206, 0.590601, 0.555796, 0.881451, 0.529031, 
            0.573261, 0.213237, 0.643506, 0.426401, 0.271222, 
            0.210728, 0.649544, 0.613671, 0.266659, 0.494369, 
            0.448391, 0.824189, 0.486635, 0.455239, 0.259476
        };
        x_sigmoid_b = {
            0.132815, 0.009581, 0.177470, -0.035969, 0.064459, 
            -0.073499, 0.012434, -0.036041, -0.384974, -0.241164, 
            0.206977, -0.356955, 0.126292, 0.214207, 0.010142, 
            -0.081242, 0.077336, 0.065731, -0.041045, -0.380001, 
            0.015473, -0.066180, 0.046631, -0.126118, -0.043483, 
            -0.185469, 0.103964, -0.030637, 0.031639, 0.538344
        };
        x_dim = x.size();

        cudaMalloc(&x_g, x_dim * sizeof(double));
        cudaMalloc(&g_g, x_dim * sizeof(double));
        cudaMalloc(&x_sigmoid_f_g, x_dim * sizeof(double));
        cudaMalloc(&x_sigmoid_b_g, x_dim * sizeof(double));

        cudaMemcpy(x_g, x.data(), x_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(g_g, g.data(), x_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(x_sigmoid_f_g, x_sigmoid_f.data(), x_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(x_sigmoid_b_g, x_sigmoid_b.data(), x_dim * sizeof(double), cudaMemcpyHostToDevice);
    }
    void TearDown() override {
        cudaFree(x_g);
        cudaFree(g_g);
        cudaFree(x_sigmoid_f_g);
        cudaFree(x_sigmoid_b_g);
    }

};

class TestSoftmax : public ::testing::Test {
protected:
    std::vector<double> y;
    std::vector<double> y_softmax;

    double* y_g;

    int y_dim;

    void SetUp() override {
        y = {
            -1.028684, 0.856440, 1.369762, -1.437391, 1.551560, 1.139737, -1.240337, -0.648702, -0.400014, 1.586942, 
            2.365771, 2.535360, -0.772002, 0.039393, -1.142135, 1.507503, 0.550930, 0.630071, -0.746441, 0.497415, 
            -0.382562, -1.579024, 1.228670, -0.061057, -0.585326, -1.225693, -0.035275, 0.099546, 0.465645, 0.714231, 
            -0.739603, 0.209539, 0.564118, 0.357420, -0.649761, 1.078385, -0.351789, -1.801129, -0.612122, -0.219620, 
            0.764168, -1.062313, 0.094680, -0.484254, -1.003578, 0.560764, -0.030785, 0.453219, 0.187955, 0.185473
        };
        y_softmax = {
            0.016942, 0.111600, 0.186465, 0.011258, 0.223640, 0.148149, 0.013710, 0.024774, 0.031768, 0.231695, 
            0.301412, 0.357118, 0.013075, 0.029432, 0.009030, 0.127767, 0.049089, 0.053132, 0.013414, 0.046531, 
            0.057797, 0.017470, 0.289503, 0.079714, 0.047189, 0.024874, 0.081795, 0.093601, 0.134982, 0.173075, 
            0.045141, 0.116621, 0.166253, 0.135208, 0.049384, 0.278044, 0.066527, 0.015616, 0.051279, 0.075927, 
            0.190346, 0.030642, 0.097452, 0.054621, 0.032495, 0.155313, 0.085961, 0.139477, 0.106979, 0.106714
        };
        y_dim = y.size();

        cudaMalloc(&y_g, y_dim * sizeof(double));

        cudaMemcpy(y_g, y.data(), y_dim * sizeof(double), cudaMemcpyHostToDevice);
    }
    void TearDown() override {
        cudaFree(y_g);
    }
};

TEST_F(TestReLU, relu_forward) {
    double* vt_relu_f_g;
    cudaMalloc(&vt_relu_f_g, v_dim * sizeof(double));
    ops::relu_forward<double, device::GPU>()(device::gpu_device, vt_relu_f_g, v_g, v_dim);

    double* vt_relu_f = new double[v_dim];
    cudaMemcpy(vt_relu_f, vt_relu_f_g, v_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < v_dim; i++) {
        EXPECT_NEAR(vt_relu_f[i], v_relu_f[i], 1e-6);
    }
}

TEST_F(TestReLU, relu_backward) {
    double* vt_relu_b_g;
    cudaMalloc(&vt_relu_b_g, v_dim * sizeof(double));
    ops::relu_backward<double, device::GPU>()(device::gpu_device, vt_relu_b_g, v_g, g_g, v_dim);

    double* vt_relu_b = new double[v_dim];
    cudaMemcpy(vt_relu_b, vt_relu_b_g, v_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < v_dim; i++) {
        EXPECT_NEAR(vt_relu_b[i], v_relu_b[i], 1e-6);
    }
}

TEST_F(TestSigmoid, sigmoid_forward) {
    double* xt_sigmoid_f_g;
    cudaMalloc(&xt_sigmoid_f_g, x_dim * sizeof(double));
    ops::sigmoid_forward<double, device::GPU>()(device::gpu_device, xt_sigmoid_f_g, x_g, x_dim);

    double* xt_sigmoid_f = new double[x_dim];
    cudaMemcpy(xt_sigmoid_f, xt_sigmoid_f_g, x_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < x_dim; i++) {
        EXPECT_NEAR(xt_sigmoid_f[i], x_sigmoid_f[i], 1e-6);
    }
}

TEST_F(TestSigmoid, sigmoid_backward) {
    double* xt_sigmoid_b_g;
    cudaMalloc(&xt_sigmoid_b_g, x_dim * sizeof(double));
    ops::sigmoid_backward<double, device::GPU>()(device::gpu_device, xt_sigmoid_b_g, x_g, g_g, x_dim);

    double* xt_sigmoid_b = new double[x_dim];
    cudaMemcpy(xt_sigmoid_b, xt_sigmoid_b_g, x_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < x_dim; i++) {
        EXPECT_NEAR(xt_sigmoid_b[i], x_sigmoid_b[i], 1e-6);
    }
}

TEST_F(TestSoftmax, softmax_forward) {
    double* yt_softmax_g;
    cudaMalloc(&yt_softmax_g, y_dim * sizeof(double));
    ops::softmax_forward<double, device::GPU>()(device::gpu_device, yt_softmax_g, y_g, 5, 10);

    double* yt_softmax = new double[y_dim];
    cudaMemcpy(yt_softmax, yt_softmax_g, y_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < y_dim; i++) {
        EXPECT_NEAR(yt_softmax[i], y_softmax[i], 1e-6);
    }
}


int main(int argc, char** argv) {
    std::cout << "run test for CORE::KERNELS::ACTIVATION::GPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

