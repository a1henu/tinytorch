/**
 * @file test_activation_cpu.cpp
 * @brief Activation function test cases for CPU
 * 
 * @copyright Copyright (c) 2024 chenxu bai
 * Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "core/device/device.h"
#include "core/kernels/activation/relu.h"
#include "core/kernels/activation/sigmoid.h"

class TestReLU : public ::testing::Test {
protected:
    std::vector<double> v;
    std::vector<double> g;

    std::vector<double> v_relu_f;
    std::vector<double> v_relu_b;

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
    }
    void TearDown() override {
    }

};

class TestSigmoid : public ::testing::Test {
protected:
    std::vector<double> x;
    std::vector<double> g;

    std::vector<double> x_sigmoid_f;
    std::vector<double> x_sigmoid_b;

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
    }
    void TearDown() override {
    }

};

TEST_F(TestReLU, forward) {
    std::vector<double> vt_relu_f(v_dim);
    activation::relu_forward<double, device::CPU>()(device::cpu_device, vt_relu_f.data(), v.data(), v_dim);
    for (int i = 0; i < v_dim; i++) {
        EXPECT_NEAR(vt_relu_f[i], v_relu_f[i], 1e-6);
    }
}

TEST_F(TestReLU, backward) {
    std::vector<double> vt_relu_b(v_dim);
    activation::relu_backward<double, device::CPU>()(device::cpu_device, vt_relu_b.data(), v.data(), g.data(), v_dim);
    for (int i = 0; i < v_dim; i++) {
        EXPECT_NEAR(vt_relu_b[i], v_relu_b[i], 1e-6);
    }
}

TEST_F(TestSigmoid, forward) {
    std::vector<double> xt_sigmoid_f(x_dim);
    activation::sigmoid_forward<double, device::CPU>()(device::cpu_device, xt_sigmoid_f.data(), x.data(), x_dim);
    for (int i = 0; i < x_dim; i++) {
        EXPECT_NEAR(xt_sigmoid_f[i], x_sigmoid_f[i], 1e-6);
    }
}

TEST_F(TestSigmoid, backward) {
    std::vector<double> xt_sigmoid_b(x_dim);
    activation::sigmoid_backward<double, device::CPU>()(device::cpu_device, xt_sigmoid_b.data(), x.data(), g.data(), x_dim);
    for (int i = 0; i < x_dim; i++) {
        EXPECT_NEAR(xt_sigmoid_b[i], x_sigmoid_b[i], 1e-6);
    }
}

int main(int argc, char** argv) {
    std::cout << "run test for CORE::KERNELS::ACTIVATION::CPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}