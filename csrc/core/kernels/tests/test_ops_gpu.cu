/**
 * @file test_ops_gpu.cu
 * @brief Math operator test cases for GPU
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
    std::vector<double> v1;
    std::vector<double> v2;

    double* vt_1;
    double* vt_2;

    int vt_dim;

    void SetUp() override {
        v1 = generate_random_vector(100, 0.0, 1.0); 
        v2 = generate_random_vector(100, 0.0, 1.0);
        vt_dim = v1.size();

        cudaMalloc(&vt_1, vt_dim * sizeof(double));
        cudaMalloc(&vt_2, vt_dim * sizeof(double));

        cudaMemcpy(vt_1, v1.data(), vt_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(vt_2, v2.data(), vt_dim * sizeof(double), cudaMemcpyHostToDevice);
    }
    void TearDown() override {
        cudaFree(vt_1);
        cudaFree(vt_2);
    }
    
    using add_cpu_op = ops::add_op<double, device::CPU>;
    using sub_cpu_op = ops::sub_op<double, device::CPU>;
    using smatmul_cpu_op = ops::matmul_op<float, device::CPU>;
    using dmatmul_cpu_op = ops::matmul_op<double, device::CPU>;
    using equal_cpu_op = ops::equal_op<double, device::CPU>;
    using ones_cpu_op = ops::ones_op<double, device::CPU>;
    using eye_cpu_op = ops::eye_op<double, device::CPU>;
    using im2col_cpu_op = ops::im2col_op<int, device::CPU>;
    using col2im_cpu_op = ops::col2im_op<int, device::CPU>;
    using conv2d_cpu_op = ops::conv2d_forward_op<double, device::CPU>;
    using conv2d_backward_cpu_op = ops::conv2d_backward_op<double, device::CPU>;
    using max_pool_cpu_op = ops::max_pool_forward_op<double, device::CPU>;
    using max_pool_backward_cpu_op = ops::max_pool_backward_op<double, device::CPU>;

    using add_gpu_op = ops::add_op<double, device::GPU>;
    using sub_gpu_op = ops::sub_op<double, device::GPU>;
    using smatmul_gpu_op = ops::matmul_op<float, device::GPU>;
    using dmatmul_gpu_op = ops::matmul_op<double, device::GPU>;
    using equal_gpu_op = ops::equal_op<double, device::GPU>;
    using ones_gpu_op = ops::ones_op<double, device::GPU>;
    using eye_gpu_op = ops::eye_op<double, device::GPU>;
    using im2col_gpu_op = ops::im2col_op<int, device::GPU>;
    using col2im_gpu_op = ops::col2im_op<int, device::GPU>;
    using conv2d_gpu_op = ops::conv2d_forward_op<double, device::GPU>;
    using conv2d_backward_gpu_op = ops::conv2d_backward_op<double, device::GPU>;
    using max_pool_gpu_op = ops::max_pool_forward_op<double, device::GPU>;
    using max_pool_backward_gpu_op = ops::max_pool_backward_op<double, device::GPU>;
};

TEST_F(TestOps, TestAddOp_gpu_1) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    add_gpu_op()(device::gpu_device, vt_out, vt_1, vt_2, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], v1[i] + v2[i]);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestAddOp_gpu_2) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    add_gpu_op()(device::gpu_device, vt_out, vt_2, vt_1, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], v1[i] + v2[i]);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestSubOp_gpu) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    sub_gpu_op()(device::gpu_device, vt_out, vt_1, vt_2, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], v1[i] - v2[i]);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestMatmulOp_gpu_float) {
    const int m = 30, n = 40, k = 35;
    std::vector<float> A = generate_random_vector(m * k, 0.0f, 1.0f);
    std::vector<float> B = generate_random_vector(k * n, 0.0f, 1.0f);
    float* d_A;
    float* d_B;
    float* d_C;
    float* h_C = new float[m * n];

    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    smatmul_gpu_op()(device::gpu_device, "N", "N", m, n, k, alpha, d_A, k, d_B, n, beta, d_C, n);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> C_expected(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_C[i], C_expected[i], 1e-4f);
    }

    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(TestOps, TestMatmulOp_gpu_double) {
    const int m = 2, n = 3, k = 4;
    std::vector<double> A = generate_random_vector(m * k, 0.0, 1.0);
    std::vector<double> B = generate_random_vector(k * n, 0.0, 1.0);
    double* d_A;
    double* d_B;
    double* d_C;
    double* h_C = new double[m * n];

    cudaMalloc(&d_A, m * k * sizeof(double));
    cudaMalloc(&d_B, k * n * sizeof(double));
    cudaMalloc(&d_C, m * n * sizeof(double));

    cudaMemcpy(d_A, A.data(), m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(double), cudaMemcpyHostToDevice);

    const double alpha = 1.0;
    const double beta = 0.0;

    dmatmul_gpu_op()(device::gpu_device, "N", "N", m, n, k, alpha, d_A, k, d_B, n, beta, d_C, n);

    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> C_expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C_expected[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }

    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_C[i], C_expected[i], 1e-4);
    }

    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(TestOps, TestEqualOp_gpu_1) {
    bool vt_out_c = true;
    bool* vt_out_g;
    cudaMalloc(&vt_out_g, sizeof(bool));
    equal_gpu_op()(device::gpu_device, vt_out_g, vt_1, vt_2, vt_dim);
    cudaMemcpy(&vt_out_c, vt_out_g, sizeof(bool), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        if (v1[i] != v2[i]) {
            EXPECT_FALSE(vt_out_c);
            cudaFree(vt_out_g);
            return;
        }
    }
    EXPECT_TRUE(vt_out_c);
    cudaFree(vt_out_g);
}

TEST_F(TestOps, TestEqualOp_gpu_2) {
    double* vt_out1;
    double* vt_out2;
    bool* vt_out_g;
    bool* vt_out_c = new bool;
    cudaMalloc(&vt_out1, vt_dim * sizeof(double));
    cudaMalloc(&vt_out2, vt_dim * sizeof(double));
    cudaMalloc(&vt_out_g, sizeof(bool));
    add_gpu_op()(device::gpu_device, vt_out1, vt_1, vt_2, vt_dim);
    add_gpu_op()(device::gpu_device, vt_out2, vt_2, vt_1, vt_dim);
    equal_gpu_op()(device::gpu_device, vt_out_g, vt_out1, vt_out2, vt_dim);
    cudaMemcpy(vt_out_c, vt_out_g, sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(*vt_out_c);
    cudaFree(vt_out1);
    cudaFree(vt_out2);
    cudaFree(vt_out_g);
    delete vt_out_c;
}

TEST_F(TestOps, TestOnesOp_gpu) {
    double* vt_out;
    double* vt_out_cpu = new double[vt_dim];
    cudaMalloc(&vt_out, vt_dim * sizeof(double));
    ones_gpu_op()(device::gpu_device, vt_out, vt_dim);
    cudaMemcpy(vt_out_cpu, vt_out, vt_dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vt_dim; ++i) {
        EXPECT_EQ(vt_out_cpu[i], 1.0);
    }
    delete[] vt_out_cpu;
    cudaFree(vt_out);
}

TEST_F(TestOps, TestEyeOp_gpu) {
    const int dim = 100;
    double* vt_out;
    double* vt_out_cpu = new double[dim * dim];
    cudaMalloc(&vt_out, dim * dim * sizeof(double));
    eye_gpu_op()(device::gpu_device, vt_out, dim);
    cudaMemcpy(vt_out_cpu, vt_out, dim * dim * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i == j) {
                EXPECT_EQ(vt_out_cpu[i * dim + j], 1.0);
            } else {
                EXPECT_EQ(vt_out_cpu[i * dim + j], 0.0);
            }
        }
    }
}

TEST_F(TestOps, TestIm2ColOp_gpu) {
    /**
     * example img is 
     * [
     * [1, 2, 3;    // channel 1
     *  4, 5, 6;
     *  7, 8, 9],
     * [3, 2, 1;    // channel 2
     *  6, 5, 4;
     *  9, 8, 7]
     * ]
     * 
     * im2col output format (channels_col = 8, height_out = width_out = 2):
     * c=0 (c_im=0, h_off=0, w_off=0): [1,2,4,5]
     * c=1 (c_im=0, h_off=0, w_off=1): [2,3,5,6]
     * c=2 (c_im=0, h_off=1, w_off=0): [4,5,7,8]
     * c=3 (c_im=0, h_off=1, w_off=1): [5,6,8,9]
     * c=4 (c_im=1, h_off=0, w_off=0): [3,2,6,5]
     * c=5 (c_im=1, h_off=0, w_off=1): [2,1,5,4]
     * c=6 (c_im=1, h_off=1, w_off=0): [6,5,9,8]
     * c=7 (c_im=1, h_off=1, w_off=1): [5,4,8,7]
     */
    int data_im[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7};
    int data_col[32] = {0};
    
    int gt_col[32] = {
        // c=0: channel1, kernel(0,0)
        1, 2, 4, 5,
        // c=1: channel1, kernel(0,1)
        2, 3, 5, 6,
        // c=2: channel1, kernel(1,0)
        4, 5, 7, 8,
        // c=3: channel1, kernel(1,1)
        5, 6, 8, 9,
        // c=4: channel2, kernel(0,0)
        3, 2, 6, 5,
        // c=5: channel2, kernel(0,1)
        2, 1, 5, 4,
        // c=6: channel2, kernel(1,0)
        6, 5, 9, 8,
        // c=7: channel2, kernel(1,1)
        5, 4, 8, 7
    };

    int* gdata_im;
    int* gdata_col;
    cudaMalloc(&gdata_im, 18 * sizeof(int));
    cudaMalloc(&gdata_col, 32 * sizeof(int));

    cudaMemcpy(gdata_im, data_im, 18 * sizeof(int), cudaMemcpyHostToDevice);
    im2col_gpu_op()(device::gpu_device, gdata_im, gdata_col, 2, 3, 3, 2, 2, 0, 0, 1, 1);
    cudaMemcpy(data_col, gdata_col, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(data_col[i], gt_col[i]);
    }
}

TEST_F(TestOps, TestCol2ImOp_gpu) {
    /**
     * the col img is 
     * c=0 (c_im=0, h_off=0, w_off=0): [1,2,4,5]
     * c=1 (c_im=0, h_off=0, w_off=1): [2,3,5,6]
     * c=2 (c_im=0, h_off=1, w_off=0): [4,5,7,8]
     * c=3 (c_im=0, h_off=1, w_off=1): [5,6,8,9]
     * c=4 (c_im=1, h_off=0, w_off=0): [3,2,6,5]
     * c=5 (c_im=1, h_off=0, w_off=1): [2,1,5,4]
     * c=6 (c_im=1, h_off=1, w_off=0): [6,5,9,8]
     * c=7 (c_im=1, h_off=1, w_off=1): [5,4,8,7]
     * 
     * col2im output format (height = width = 3):
     * [
     * [1, 2, 3;    // channel 1
     *  4, 5, 6;
     *  7, 8, 9],
     * [3, 2, 1;    // channel 2
     *  6, 5, 4;
     *  9, 8, 7]
     * ]
     * 
     * so data_im is
     * [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7]
     */
    int data_col[32] = {
        1, 2, 4, 5, 
        2, 3, 5, 6, 
        4, 5, 7, 8, 
        5, 6, 8, 9, 
        3, 2, 6, 5, 
        2, 1, 5, 4, 
        6, 5, 9, 8, 
        5, 4, 8, 7
    };

    int gt_im[18] = {
        1, 4, 3, 
        8, 20, 12, 
        7, 16, 9, 
        3, 4, 1, 
        12, 20, 8, 
        9, 16, 7
    };
    int data_im[18] = {0};

    int* gdata_col;
    int* gdata_im;
    cudaMalloc(&gdata_col, 32 * sizeof(int));
    cudaMalloc(&gdata_im, 18 * sizeof(int));
    cudaMemcpy(gdata_col, data_col, 32 * sizeof(int), cudaMemcpyHostToDevice);

    col2im_gpu_op()(
        device::gpu_device,
        gdata_col,
        gdata_im,
        2, 
        3, 3, 
        2, 2, 
        0, 0, 
        1, 1
    );
    cudaMemcpy(data_im, gdata_im, 18 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 18; ++i) {
        EXPECT_EQ(data_im[i], gt_im[i]);
    }
}

TEST_F(TestOps, TestConv2dForwardOp_gpu) {
    /**
     * example input is (2, 2, 4, 4)
     * weight is (3, 2, 3, 3)
     * bias is (3)
     * stride = 1, padding = 1
     * output shape should be (2, 3, 4, 4)
     */
    double input[64] = {
        1.9269152879714966, 1.4872840642929077, 0.9007171988487244, -2.1055209636688232, 
        0.6784184575080872, -1.2345448732376099, -0.04306747764348984, -1.6046669483184814, 
        -0.7521352767944336, 1.6487230062484741, -0.3924786448478699, -1.4036071300506592, 
        -0.7278813123703003, -0.5594301819801331, -0.7688388824462891, 0.7624453902244568, 
        1.6423169374465942, -0.1595974713563919, -0.4973975419998169, 0.439589262008667, 
        -0.7581311464309692, 1.078317642211914, 0.8008005619049072, 1.680620551109314, 
        1.27912437915802, 1.2964228391647339, 0.610466480255127, 1.334737777709961, 
        -0.2316243201494217, 0.041759490966796875, -0.2515752911567688, 0.859858512878418, 
        -1.3846737146377563, -0.8712361454963684, -0.223365917801857, 1.7173614501953125, 
        0.3188803195953369, -0.42451897263526917, 0.3057209253311157, -0.7745925188064575, 
        -1.5575724840164185, 0.9956361055374146, -0.8797858357429504, -0.6011420488357544, 
        -1.2741512060165405, 2.1227850914001465, -1.234653115272522, -0.4879138767719269, 
        -0.9138230085372925, -0.6581372618675232, 0.07802387326955795, 0.5258087515830994, 
        -0.48799172043800354, 1.1913690567016602, -0.8140076398849487, -0.7359927892684937, 
        -1.4032478332519531, 0.03600366786122322, -0.06347727030515671, 0.6756148934364319, 
        -0.0978068932890892, 1.8445940017700195, -1.184537410736084, 1.3835493326187134
    };

    double weight[54] = {
        1.4451338052749634, 0.8564125299453735, 2.218075752258301, 0.5231655240058899, 
        0.34664666652679443, -0.19733144342899323, -1.0545889139175415, 1.2779955863952637, 
        -0.1721901297569275, 0.5237884521484375, 0.056621819734573364, 0.4262961447238922, 
        0.575005054473877, -0.6417241096496582, -2.2063984870910645, -0.7508030533790588, 
        0.01086814422160387, -0.33874234557151794, -1.3406795263290405, -0.5853705406188965, 
        0.5361881256103516, 0.5246226191520691, 1.1412016153335571, 0.05164359509944916, 
        0.7439519762992859, -0.4815843999385834, -1.0494661331176758, 0.603898823261261, 
        -1.7222950458526611, -0.827768862247467, 1.334702968597412, 0.48353928327560425, 
        -2.5095443725585938, 0.4880010485649109, 0.7845868468284607, 0.02864718623459339, 
        0.640755295753479, 0.5832474231719971, -0.3890652060508728, 0.5279164910316467, 
        1.031091570854187, -0.7047650218009949, 1.013148307800293, -0.330817848443985, 
        0.517693042755127, 0.38777846097946167, 0.7199674844741821, 0.41140761971473694, 
        -0.5733190178871155, 0.5068639516830444, -0.4752098321914673, -0.49202650785446167, 
        0.27037355303764343, -0.5628241896629333
    };

    double bias[3] = {1.2175776958465576, -0.8914098143577576, 0.7859604954719543};

    double gt_output[96] = {
        1.5963506698608398, 2.7315924167633057, 2.1339099407196045, -2.1974880695343018, 
        3.1074042320251465, 6.768407821655273, -7.819884300231934, -2.540148973464966, 
        -5.640878677368164, 0.30724698305130005, -5.273097991943359, 1.0817327499389648, 
        4.770094871520996, 1.6477982997894287, -0.9879980087280273, -0.9907224178314209, 
        2.9840078353881836, 6.869828224182129, 1.240325927734375, -0.8228897452354431, 
        -6.607905387878418, -5.0221381187438965, -3.277536630630493, -0.1672230362892151, 
        -4.188365459442139, -1.3631736040115356, -4.168606758117676, -3.0400760173797607, 
        -3.919621706008911, -3.741889238357544, -8.35329818725586, -0.9269930124282837, 
        0.9574441313743591, 3.2031567096710205, 0.28703826665878296, 0.14605748653411865, 
        3.839348316192627, 0.4381452798843384, 2.1176822185516357, -0.8986813426017761, 
        -0.4803958535194397, 1.8588266372680664, 2.176795482635498, -0.6598463654518127, 
        0.666612982749939, 2.95316743850708, 1.4654436111450195, 1.8738534450531006, 
        3.0197973251342773, -0.31642258167266846, -0.9254255294799805, 0.6943027377128601, 
        -6.543670654296875, 2.3445072174072266, 4.000932693481445, 2.5464296340942383, 
        -1.5058248043060303, 5.235692977905273, -7.281713485717773, 0.9721486568450928, 
        -2.8397738933563232, -1.2056043148040771, 0.41983914375305176, -2.9476840496063232, 
        -1.3636322021484375, -3.5632517337799072, -3.403423309326172, 0.9354375004768372, 
        -2.7078466415405273, 2.165151357650757, 5.273238658905029, -4.487553596496582, 
        -5.584446907043457, -3.3107407093048096, 2.3195712566375732, -1.1535500288009644, 
        -3.0796196460723877, 4.709719657897949, -4.365135669708252, -2.678490400314331, 
        -1.3059120178222656, 1.1094567775726318, -2.271902322769165, 3.4280896186828613, 
        -0.16583764553070068, -2.159982681274414, -0.08886110782623291, 1.0310784578323364, 
        -1.3043057918548584, 1.2158626317977905, 0.5350228548049927, -1.1534976959228516, 
        -5.241312503814697, 4.106808662414551, -0.7330557703971863, 0.5589356422424316
    };
    double output[96] = {0};

    double* d_input;
    double* d_weight;
    double* d_bias;
    double* d_output;
    cudaMalloc(&d_input, 64 * sizeof(double));
    cudaMalloc(&d_weight, 54 * sizeof(double));
    cudaMalloc(&d_bias, 3 * sizeof(double));
    cudaMalloc(&d_output, 96 * sizeof(double));
    cudaMemcpy(d_input, input, 64 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, 54 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, 3 * sizeof(double), cudaMemcpyHostToDevice);
    conv2d_gpu_op()(
        device::gpu_device,
        d_output,
        d_input,
        d_weight,
        d_bias,
        2, 2, 3, 
        4, 4, 
        3, 3, 
        1, 1, 
        1, 1
    );
    cudaMemcpy(output, d_output, 96 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 96; ++i) {
        EXPECT_NEAR(output[i], gt_output[i], 1e-4);
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}


TEST_F(TestOps, TestConv2dBackwardOp_gpu) {
    /**
     * example input is (2, 2, 4, 4)
     * weight is (3, 2, 3, 3)
     * bias is (3)
     * stride = 1, padding = 1
     * output shape should be (2, 3, 4, 4)
     */
    double input[64] = {
        1.9269152879714966, 1.4872840642929077, 0.9007171988487244, -2.1055209636688232, 
        0.6784184575080872, -1.2345448732376099, -0.04306747764348984, -1.6046669483184814, 
        -0.7521352767944336, 1.6487230062484741, -0.3924786448478699, -1.4036071300506592, 
        -0.7278813123703003, -0.5594301819801331, -0.7688388824462891, 0.7624453902244568, 
        1.6423169374465942, -0.1595974713563919, -0.4973975419998169, 0.439589262008667, 
        -0.7581311464309692, 1.078317642211914, 0.8008005619049072, 1.680620551109314, 
        1.27912437915802, 1.2964228391647339, 0.610466480255127, 1.334737777709961, 
        -0.2316243201494217, 0.041759490966796875, -0.2515752911567688, 0.859858512878418, 
        -1.3846737146377563, -0.8712361454963684, -0.223365917801857, 1.7173614501953125, 
        0.3188803195953369, -0.42451897263526917, 0.3057209253311157, -0.7745925188064575, 
        -1.5575724840164185, 0.9956361055374146, -0.8797858357429504, -0.6011420488357544, 
        -1.2741512060165405, 2.1227850914001465, -1.234653115272522, -0.4879138767719269, 
        -0.9138230085372925, -0.6581372618675232, 0.07802387326955795, 0.5258087515830994, 
        -0.48799172043800354, 1.1913690567016602, -0.8140076398849487, -0.7359927892684937, 
        -1.4032478332519531, 0.03600366786122322, -0.06347727030515671, 0.6756148934364319, 
        -0.0978068932890892, 1.8445940017700195, -1.184537410736084, 1.3835493326187134
    };

    double weight[54] = {
        1.4451338052749634, 0.8564125299453735, 2.218075752258301, 0.5231655240058899, 
        0.34664666652679443, -0.19733144342899323, -1.0545889139175415, 1.2779955863952637, 
        -0.1721901297569275, 0.5237884521484375, 0.056621819734573364, 0.4262961447238922, 
        0.575005054473877, -0.6417241096496582, -2.2063984870910645, -0.7508030533790588, 
        0.01086814422160387, -0.33874234557151794, -1.3406795263290405, -0.5853705406188965, 
        0.5361881256103516, 0.5246226191520691, 1.1412016153335571, 0.05164359509944916, 
        0.7439519762992859, -0.4815843999385834, -1.0494661331176758, 0.603898823261261, 
        -1.7222950458526611, -0.827768862247467, 1.334702968597412, 0.48353928327560425, 
        -2.5095443725585938, 0.4880010485649109, 0.7845868468284607, 0.02864718623459339, 
        0.640755295753479, 0.5832474231719971, -0.3890652060508728, 0.5279164910316467, 
        1.031091570854187, -0.7047650218009949, 1.013148307800293, -0.330817848443985, 
        0.517693042755127, 0.38777846097946167, 0.7199674844741821, 0.41140761971473694, 
        -0.5733190178871155, 0.5068639516830444, -0.4752098321914673, -0.49202650785446167, 
        0.27037355303764343, -0.5628241896629333
    };

    double dy[96] = {
        -1.4241265058517456, -0.11632604151964188, -0.9726738929748535, 0.9584577679634094, 
        1.6192004680633545, 1.450609803199768, 0.2694815397262573, -0.21037597954273224, 
        -1.4391242265701294, 0.521381676197052, 0.3487516939640045, 0.9675941467285156, 
        -0.46568843722343445, 1.6047972440719604, -2.4801201820373535, -0.4175437390804291, 
        -0.19333650171756744, 0.652643620967865, -1.9005532264709473, 0.22857652604579926, 
        0.02485940419137478, -0.34595024585723877, 0.2868328094482422, -0.7308424115180969, 
        1.4611194133758545, -0.3097529113292694, -1.6021603345870972, 1.3528969287872314, 
        1.288827657699585, 0.05229547247290611, -1.5468504428863525, 0.7567060589790344, 
        -0.5854812264442444, -0.3899965286254883, 0.03581761196255684, 0.12058872729539871, 
        -0.8056637048721313, -0.20757682621479034, -0.9319478273391724, -1.5909662246704102, 
        -0.3749968111515045, 0.8032528162002563, -0.5187733173370361, -1.5012763738632202, 
        -1.9266542196273804, 0.1278512328863144, 1.0229133367538452, -0.5557951331138611, 
        -0.7499417662620544, -0.5921903252601624, 1.7743884325027466, -0.921550989151001, 
        0.9624499082565308, -0.33701515197753906, -1.1753336191177368, 0.35805708169937134, 
        -0.7060679793357849, -0.716905951499939, 0.5260620713233948, 2.1120378971099854, 
        -0.5207571387290955, -0.9320061206817627, 0.18516133725643158, 1.0686918497085571, 
        -0.5778480768203735, 0.3254614472389221, -0.8146268725395203, -1.0212392807006836, 
        -0.49492356181144714, -0.5922516584396362, 0.15431594848632812, 0.4407670795917511, 
        0.3125324845314026, -0.0335015207529068, -0.39799532294273376, 1.0804862976074219, 
        -1.7808643579483032, 1.5080454349517822, 0.30942854285240173, -0.5003090500831604, 
        0.6630072593688965, 0.7047329545021057, -0.004505051765590906, 1.666792392730713, 
        0.15392017364501953, -1.0602530241012573, -0.572657585144043, 0.0835680365562439, 
        0.4439751207828522, -0.7240339517593384, -0.07198750972747803, -0.906094491481781, 
        -2.0487122535705566, -1.0810555219650269, 0.01762307994067669, 0.0782259851694107
    };

    double gt_dx[64] = {
        2.087036609649658, 4.148775577545166, 0.835908055305481, 1.2198140621185303, 
        -1.716427206993103, 0.555419385433197, -3.721998691558838, 0.6040037274360657, 
        1.6966148614883423, 1.6706160306930542, -2.9753973484039307, -7.528209209442139, 
        -2.1279306411743164, -2.3201911449432373, -1.5841354131698608, 2.2688376903533936, 
        1.4906558990478516, 1.8013763427734375, -2.184288263320923, 6.05059814453125, 
        -3.142226457595825, -4.208440780639648, -0.8501050472259521, -3.513791561126709, 
        -3.1822400093078613, -5.842564105987549, 5.358561992645264, 1.1225804090499878, 
        0.956265389919281, -6.376407623291016, -2.640886068344116, 10.014076232910156, 
        0.8285524845123291, 0.019076434895396233, -2.9235994815826416, -2.3593766689300537, 
        -2.421774387359619, -6.9361138343811035, 5.449661731719971, 2.295773983001709, 
        -4.606927871704102, -3.5098745822906494, 1.691066026687622, 2.7424323558807373, 
        -5.7945733070373535, 0.4454790949821472, 0.1271418184041977, 2.6155993938446045, 
        0.300132691860199, 3.9963161945343018, -4.576563835144043, -2.4691805839538574, 
        -2.116002082824707, -4.001400947570801, 4.079283237457275, -0.34823256731033325, 
        2.502875804901123, -2.218743324279785, 2.859823226928711, 0.4858661890029907, 
        1.7670451402664185, 7.095067977905273, -1.562021255493164, -1.6048928499221802
    };

    double gt_dw[54] = {
        0.05605880916118622, 4.851386070251465, 3.506993055343628, 0.2207024246454239, 
        -8.72378158569336, -4.801758766174316, -3.5637032985687256, -1.779383897781372, 
        5.569955348968506, 3.3956544399261475, 3.5275051593780518, -6.101052761077881, 
        -2.3188114166259766, 2.0635030269622803, 4.021909236907959, 3.6333506107330322, 
        5.614755153656006, -1.9912424087524414, -2.600287437438965, 4.646949768066406, 
        1.8854655027389526, -6.188567161560059, 1.1853886842727661, 0.3925049602985382, 
        2.5703210830688477, -0.029653023928403854, 3.044015884399414, -5.072913646697998, 
        3.5809357166290283, -0.3303743004798889, 0.5864400863647461, 3.284113883972168, 
        -7.762607097625732, -4.713604927062988, 4.630054473876953, -4.078988075256348, 
        2.9430313110351562, 8.036064147949219, -5.158707141876221, 1.1382229328155518, 
        5.4528279304504395, 0.16640852391719818, 3.4221928119659424, -1.9640740156173706, 
        3.1793344020843506, 3.249022960662842, -2.639681816101074, -1.1722908020019531, 
        -2.863807201385498, -10.030014991760254, -3.595982551574707, -1.0769668817520142, 
        -8.193094253540039, -2.4798710346221924
    };
    double gt_db[3] = {0.5493749976158142, -2.607210636138916, -9.93616008758545};

    double dx[64] = {0};
    double dw[54] = {0};
    double db[3] = {0};

    double* d_input;
    double* d_weight;
    double* d_dy;
    double* d_dx;
    double* d_dw;
    double* d_db;

    cudaMalloc(&d_input, 64 * sizeof(double));
    cudaMalloc(&d_weight, 54 * sizeof(double));
    cudaMalloc(&d_dy, 96 * sizeof(double));
    cudaMalloc(&d_dx, 64 * sizeof(double));
    cudaMalloc(&d_dw, 54 * sizeof(double));
    cudaMalloc(&d_db, 3 * sizeof(double));

    cudaMemcpy(d_input, input, 64 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, 54 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy, 96 * sizeof(double), cudaMemcpyHostToDevice);

    conv2d_backward_gpu_op()(
        device::gpu_device,
        d_dx, d_dw, d_db, 
        d_dy, d_input, d_weight, 
        2, 2, 3, 
        4, 4, 
        3, 3, 
        1, 1, 
        1, 1
    );

    cudaMemcpy(dx, d_dx, 64 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dw, d_dw, 54 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(db, d_db, 3 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_NEAR(dx[i], gt_dx[i], 1e-4);
    }
    for (int i = 0; i < 54; ++i) {
        EXPECT_NEAR(dw[i], gt_dw[i], 1e-4);
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(db[i], gt_db[i], 1e-4);
    }
}

TEST_F(TestOps, TestMaxPoolOp_gpu) {
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
    double gt_out[8] = {
        1.1023, 1.4135, 
        0.6764, 0.4086, 
        1.3151, 0.7686, 
        1.2346, 0.4868
    };
    double gt_mask[8] = {1, 2, 2, 0, 1, 1, 3, 0};

    double* d_data_im;
    double* d_data_out;
    double* d_mask_out;

    double data_out[8] = {0.0};
    double mask_out[8] = {0};
    
    cudaMalloc(&d_data_im, 32 * sizeof(double));
    cudaMalloc(&d_data_out, 8 * sizeof(double));
    cudaMalloc(&d_mask_out, 8 * sizeof(double));

    cudaMemcpy(d_data_im, data_im, 32 * sizeof(double), cudaMemcpyHostToDevice);
    max_pool_gpu_op()(
        device::gpu_device, 
        d_data_out, 
        d_mask_out, 
        d_data_im, 
        1, 2, 
        4, 4, 
        2, 2, 
        0, 0, 
        2, 2
    );
    cudaMemcpy(data_out, d_data_out, 8 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(mask_out, d_mask_out, 8 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(data_out[i], gt_out[i], 1e-4);
        EXPECT_EQ(mask_out[i], gt_mask[i]);
    }

    cudaFree(d_data_im);
    cudaFree(d_data_out);
    cudaFree(d_mask_out);
}

TEST_F(TestOps, TestMaxPoolBackwardOp_gpu) {
    double mask_out[8] = {1, 2, 2, 0, 1, 1, 3, 0};
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

    double* d_mask_out;
    double* d_grad_out;
    double* d_grad_im;
    
    cudaMalloc(&d_mask_out, 8 * sizeof(double));
    cudaMalloc(&d_grad_out, 8 * sizeof(double));
    cudaMalloc(&d_grad_im, 32 * sizeof(double));

    cudaMemcpy(d_mask_out, mask_out, 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_out, grad_out, 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_grad_im, 0, 32 * sizeof(double));  // 初始化为0

    max_pool_backward_gpu_op()(
        device::gpu_device, 
        d_grad_im,  
        d_mask_out,   
        d_grad_out,  
        1, 2,           
        4, 4,         
        2, 2,         
        0, 0,         
        2, 2          
    );

    cudaMemcpy(grad_im, d_grad_im, 32 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(grad_im[i], gt_grad_im[i], 1e-4);
    }

    cudaFree(d_mask_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_im);
}

int main(int argc, char** argv) {
std::cout << "run test for CORE::KERNELS::OPS::GPU" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
