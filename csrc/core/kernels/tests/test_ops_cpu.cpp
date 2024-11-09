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
    using im2col_cpu_op = ops::im2col_op<int, device::CPU>;
    using max_pool_cpu_op = ops::max_pool_forward_op<double, device::CPU>;
    using max_pool_backward_cpu_op = ops::max_pool_backward_op<double, device::CPU>;

    using add_gpu_op = ops::add_op<double, device::GPU>;
    using sub_gpu_op = ops::sub_op<double, device::GPU>;
    using smatmul_gpu_op = ops::matmul_op<float, device::GPU>;
    using dmatmul_gpu_op = ops::matmul_op<double, device::GPU>;
    using equal_gpu_op = ops::equal_op<double, device::GPU>;
    using ones_gpu_op = ops::ones_op<double, device::GPU>;
    using eye_gpu_op = ops::eye_op<double, device::GPU>;
    using trans_gpu_op = ops::transpose_op<double, device::GPU>;
    using im2col_gpu_op = ops::im2col_op<int, device::GPU>;
    using max_pool_gpu_op = ops::max_pool_forward_op<double, device::GPU>;
    using max_pool_backward_gpu_op = ops::max_pool_backward_op<double, device::GPU>;
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

    smatmul_cpu_op()(device::cpu_device, "N", "N", m, n, k, alpha, A.data(), k, B.data(), n, beta, C.data(), n);

    std::vector<float> C_expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i * n + j] += A[i * k + p] * B[p * n + j];
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

    dmatmul_cpu_op()(device::cpu_device, "N", "N", m, n, k, alpha, A.data(), k, B.data(), n, beta, C.data(), n);

    std::vector<double> C_expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i * n + j] += A[i * k + p] * B[p * n + j];
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
                EXPECT_EQ(vt_out[i * dim + j], 1.0);
            } else {
                EXPECT_EQ(vt_out[i * dim + j], 0.0);
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
            At[j * m + i] = A[i * n + j];
        }
    }

    std::vector<double> At_out(n * m, 0.0);
    trans_cpu_op()(device::cpu_device, A.data(), At_out.data(), m, n);

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(At_out[i], At[i], 1e-6);
    }
}

TEST_F(TestOps, TestIm2ColOp_cpu) {
    /**
     * example img is 
     * [
     * [1, 2, 3;
     *  4, 5, 6;
     *  7, 8, 9],
     * [3, 2, 1;
     *  6, 5, 4;
     *  9, 8, 7]
     * ]
     * 
     * so data_im is
     * [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7]
     * 
     * im2col(img)(2X2 kernel, 1X1 stride, 0 padding) is
     * [
     *  1, 2, 4, 5, 3, 2, 6, 5;
     *  2, 3, 5, 6, 2, 1, 5, 4;
     *  4, 5, 7, 8, 6, 5, 9, 8;
     *  5, 6, 8, 9, 5, 4, 8, 7
     * ]
     * 
     * so data_col is
     * [1, 2, 4, 5, 3, 2, 6, 5, 2, 3, 5, 6, 2, 1, 5, 4, 4, 5, 7, 8, 6, 5, 9, 8, 5, 6, 8, 9, 5, 4, 8, 7]
     */
    int data_im[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7};
    int data_col[32] = {0};
    int gt_col[32] = {1, 2, 4, 5, 3, 2, 6, 5, 2, 3, 5, 6, 2, 1, 5, 4, 4, 5, 7, 8, 6, 5, 9, 8, 5, 6, 8, 9, 5, 4, 8, 7};

    im2col_cpu_op()(
        device::cpu_device, 
        data_im, 
        data_col, 
        2, 
        3, 3, 
        2, 2, 
        0, 0, 
        1, 1
    );

    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(data_col[i], gt_col[i]);
    }
}

TEST_F(TestOps, TestMaxPoolingOp_cpu) {
    /**
     * example img is 
     * [
     *  [[-0.5752,  1.1023,  0.8327, -0.3337],
     *   [-0.0532,  0.8745,  1.4135, -0.4422],
     *   [-0.4538,  0.2952,  0.4086, -0.3135],
     *   [ 0.6764,  0.3422, -0.1896,  0.3065]],
     *  [[-0.3942,  1.3151,  0.5020,  0.7686],
     *   [-1.7310,  0.8545, -1.3705, -0.3178],
     *   [-2.5553,  1.1632,  0.4868, -0.1809],
     *   [ 0.0281,  1.2346,  0.3800,  0.2100]]
     * ]
     * 
     * so data_im is
     * [-0.5752, 1.1023, 0.8327, -0.3337, 
     *  -0.0532, 0.8745, 1.4135, -0.4422, 
     *  -0.4538, 0.2952, 0.4086, -0.3135, 
     *  0.6764, 0.3422, -0.1896, 0.3065, 
     *  -0.3942, 1.3151, 0.5020, 0.7686, 
     *  -1.7310, 0.8545, -1.3705, -0.3178, 
     *  -2.5553, 1.1632, 0.4868, -0.1809,  
     *  0.0281, 1.2346, 0.3800, 0.2100]
     * 
     * max_pool(img)(2X2 kernel, 2X2 stride, 0 padding) is
     * [
     * [[1.1023, 1.4135],
     * [0.6764, 0.4086]],
     * [[1.3151, 0.7686],
     * [1.2346, 0.4868]]
     * ]
     * 
     * so data_col is
     * [1.1023, 1.4135, 0.6764, 0.4086, 1.3151, 0.7686, 1.2346, 0.4868]
     * 
     * and mask_out is
     * [1, 2, 2, 0, 1, 1, 3, 0]
     */ 
    double data_im[32] = {
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
    double data_out[8] = {0.0};
    int mask_out[8] = {0};

    double gt_out[8] = {
        1.1023, 1.4135, 
        0.6764, 0.4086, 
        1.3151, 0.7686, 
        1.2346, 0.4868
    };
    int gt_mask[8] = {1, 2, 2, 0, 1, 1, 3, 0};

    max_pool_cpu_op()(
        device::cpu_device, 
        data_out, 
        mask_out, 
        data_im, 
        1, 2, 
        4, 4, 
        2, 2, 
        0, 0, 
        2, 2
    );

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(data_out[i], gt_out[i], 1e-4);
        EXPECT_EQ(mask_out[i], gt_mask[i]);
    }
}

TEST_F(TestOps, TestMaxPoolingBackwardOp_cpu) {
    /**
     * example img is 
     * [
     *  [[-0.5752,  1.1023,  0.8327, -0.3337],
     *   [-0.0532,  0.8745,  1.4135, -0.4422],
     *   [-0.4538,  0.2952,  0.4086, -0.3135],
     *   [ 0.6764,  0.3422, -0.1896,  0.3065]],
     *  [[-0.3942,  1.3151,  0.5020,  0.7686],
     *   [-1.7310,  0.8545, -1.3705, -0.3178],
     *   [-2.5553,  1.1632,  0.4868, -0.1809],
     *   [ 0.0281,  1.2346,  0.3800,  0.2100]]
     * ]
     * 
     * so mask_out is
     * [1, 2, 2, 0, 1, 1, 3, 0]
     *
     * if grad_out is
     * [1, 1, 1, 1, 1, 1, 1, 1]
     * 
     * then grad_im is
     * [0, 1, 0, 0, 
     *  0, 0, 1, 0, 
     *  0, 0, 1, 0, 
     *  1, 0, 0, 0, 
     *  0, 1, 0, 1, 
     *  0, 0, 0, 0, 
     *  0, 0, 1, 0, 
     *  0, 1, 0, 0]
     */
    int mask_out[8] = {1, 2, 2, 0, 1, 1, 3, 0};
    double grad_out[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    double grad_im[32] = {0};

    double gt_grad_im[32] = {
        0, 1, 0, 0, 
        0, 0, 1, 0, 
        0, 0, 1, 0, 
        1, 0, 0, 0, 
        0, 1, 0, 1, 
        0, 0, 0, 0, 
        0, 0, 1, 0, 
        0, 1, 0, 0
    };

    max_pool_backward_cpu_op()(
        device::cpu_device, 
        grad_im, 
        mask_out, 
        grad_out, 
        1, 2, 
        4, 4, 
        2, 2, 
        0, 0, 
        2, 2
    );

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(grad_im[i], gt_grad_im[i], 1e-4);
    }
}


int main(int argc, char** argv) {
std::cout << "run test for CORE::KERNELS::OPS::CPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
