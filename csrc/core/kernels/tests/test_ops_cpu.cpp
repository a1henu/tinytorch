/**
 * @file test_ops_cpu.cpp
 * @brief Math operator test cases for CPU
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
#include "core/kernels/ops.h"

#include "error/error.h"

std::vector<float> generate_random_vector(size_t size, float min_value, float max_value) {
    std::vector<float> vec(size);
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(min_value, max_value); 

    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });

    return vec;
}

std::vector<double> generate_random_vector(size_t size, double min_value, double max_value) {
    std::vector<double> vec(size);
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(min_value, max_value); 

    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });

    return vec;
}

class TestOps : public ::testing::Test {
protected:
    std::vector<double> vt_1;
    std::vector<double> vt_2;

    int vt_dim;

    void SetUp() override {
        vt_1 = generate_random_vector(100, 0.0, 1.0); 
        vt_2 = generate_random_vector(100, 0.0, 1.0);
        vt_dim = vt_1.size();
    }
    void TearDown() override {
    }
    
    using add_cpu_op = ops::add_op<double, device::CPU>;
    using sub_cpu_op = ops::sub_op<double, device::CPU>;
    using smatmul_cpu_op = ops::matmul_op<float, device::CPU>;
    using dmatmul_cpu_op = ops::matmul_op<double, device::CPU>;
    using equal_cpu_op = ops::equal_op<double, device::CPU>;
    using ones_cpu_op = ops::ones_op<double, device::CPU>;
    using eye_cpu_op = ops::eye_op<double, device::CPU>;
    using trans_cpu_op = ops::transpose_op<double, device::CPU>;

    using add_gpu_op = ops::add_op<double, device::GPU>;
    using sub_gpu_op = ops::sub_op<double, device::GPU>;
    using smatmul_gpu_op = ops::matmul_op<float, device::GPU>;
    using dmatmul_gpu_op = ops::matmul_op<double, device::GPU>;
    using equal_gpu_op = ops::equal_op<double, device::GPU>;
    using ones_gpu_op = ops::ones_op<double, device::GPU>;
    using eye_gpu_op = ops::eye_op<double, device::GPU>;
    using trans_gpu_op = ops::transpose_op<double, device::GPU>;
};

TEST_F(TestOps, TestAddOp_cpu_1) {
    std::vector<double> vt_out(vt_dim);
    add_cpu_op()(device::cpu_device, vt_out.data(), vt_1.data(), vt_2.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out[i], vt_1[i] + vt_2[i]);
    }
}

TEST_F(TestOps, TestAddOp_cpu_2) {
    std::vector<double> vt_out(vt_dim);
    add_cpu_op()(device::cpu_device, vt_out.data(), vt_2.data(), vt_1.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out[i], vt_1[i] + vt_2[i]);
    }
}

TEST_F(TestOps, TestSubOp_cpu) {
    std::vector<double> vt_out(vt_dim);
    sub_cpu_op()(device::cpu_device, vt_out.data(), vt_1.data(), vt_2.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out[i], vt_1[i] - vt_2[i]);
    }
}

TEST_F(TestOps, TestMatmulOp_cpu_float) {
    const int m = 30, n = 40, k = 35;
    std::vector<float> A = generate_random_vector(m * k, 0.0f, 1.0f);
    std::vector<float> B = generate_random_vector(k * n, 0.0f, 1.0f);
    std::vector<float> C(m * n, 0.0);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    smatmul_cpu_op()(device::cpu_device, "N", "N", m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);

    std::vector<float> C_expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i + j * m] += A[i + p * m] * B[j * k + p];
            }
        }
    }

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C[i], C_expected[i], 1e-4);
    }
}

TEST_F(TestOps, TestMatmulOp_cpu_double) {
    const int m = 30, n = 40, k = 35;
    std::vector<double> A = generate_random_vector(m * k, 0.0, 1.0);
    std::vector<double> B = generate_random_vector(k * n, 0.0, 1.0);
    std::vector<double> C(m * n, 0.0);

    const double alpha = 1.0;
    const double beta = 0.0;

    dmatmul_cpu_op()(device::cpu_device, "N", "N", m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);

    std::vector<double> C_expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i + j * m] += A[i + p * m] * B[j * k + p];
            }
        }
    }

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C[i], C_expected[i], 1e-4);
    }
}

TEST_F(TestOps, TestEqualOp_cpu_1) {
    bool vt_out;
    equal_cpu_op()(device::cpu_device, &vt_out, vt_1.data(), vt_2.data(), vt_dim);
    for (int i = 0; i < vt_dim; ++i) {
        if (vt_1[i] != vt_2[i]) {
            EXPECT_FALSE(vt_out);
            return;
        }
    }
    EXPECT_TRUE(vt_out);
}

TEST_F(TestOps, TestEqualOp_cpu_2) {
    std::vector<double> vt_out_1(vt_dim);
    std::vector<double> vt_out_2(vt_dim);
    add_cpu_op()(device::cpu_device, vt_out_1.data(), vt_1.data(), vt_2.data(), vt_dim);
    add_cpu_op()(device::cpu_device, vt_out_2.data(), vt_2.data(), vt_1.data(), vt_dim);
    bool vt_out;
    equal_cpu_op()(device::cpu_device, &vt_out, vt_out_1.data(), vt_out_2.data(), vt_dim);
    EXPECT_TRUE(vt_out);
}

TEST_F(TestOps, TestOnesOp_cpu) {
    const int size = 100;
    std::vector<double> vt_out(size);
    ones_cpu_op()(device::cpu_device, vt_out.data(), size);
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(vt_out[i], 1.0);
    }
}

TEST_F(TestOps, TestEyeOp_cpu) {
    const int dim = 100;
    std::vector<double> vt_out(dim * dim, 0.0);
    eye_cpu_op()(device::cpu_device, vt_out.data(), dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i == j) {
                EXPECT_EQ(vt_out[i + j * dim], 1.0);
            } else {
                EXPECT_EQ(vt_out[i + j * dim], 0.0);
            }
        }
    }
}

TEST_F(TestOps, TestTransposeOp_cpu) {
    const int m = 30, n = 40;
    std::vector<double> A = generate_random_vector(m * n, 0.0, 1.0);
    std::vector<double> At(n * m, 0.0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            At[j + i * n] = A[i + j * m];
        }
    }

    std::vector<double> At_out(n * m, 0.0);
    trans_cpu_op()(device::cpu_device, A.data(), At_out.data(), m, n);

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(At_out[i], At[i], 1e-6);
    }
}


int main(int argc, char** argv) {
std::cout << "run test for CORE::KERNELS::OPS::CPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
