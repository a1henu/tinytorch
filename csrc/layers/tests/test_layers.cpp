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

class TestConv2dLayer : public ::testing::Test {
protected:
    std::vector<double> x, w, b, y, dy, dx, dw, db;
    tensor::Tensor<double> input, weight, bias, output, input_grad, output_grad, weight_grad, bias_grad;
    
    int batch_size = 2, in_channels = 2, out_channels = 3;
    int height = 4, width = 4;
    int kernel_h = 3, kernel_w = 3;
    int pad_h = 1, pad_w = 1;
    int stride_h = 1, stride_w = 1;

    void SetUp() override {
        x = {
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

        w = {
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

        b = {1.2175776958465576, -0.8914098143577576, 0.7859604954719543};

        y = {
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

        input = tensor::Tensor<double>({batch_size, in_channels, height, width}, tensor::DeviceType::CPU, x.data());
        weight = tensor::Tensor<double>({out_channels, in_channels, kernel_h, kernel_w}, tensor::DeviceType::CPU, w.data());
        bias = tensor::Tensor<double>({out_channels}, tensor::DeviceType::CPU, b.data());
        output = tensor::Tensor<double>({batch_size, out_channels, height, width}, tensor::DeviceType::CPU);
        input_grad = tensor::Tensor<double>({batch_size, in_channels, height, width}, tensor::DeviceType::CPU);
        output_grad = tensor::Tensor<double>({batch_size, out_channels, height, width}, tensor::DeviceType::CPU);
        weight_grad = tensor::Tensor<double>({out_channels, in_channels, kernel_h, kernel_w}, tensor::DeviceType::CPU);
        bias_grad = tensor::Tensor<double>({out_channels}, tensor::DeviceType::CPU);

#ifdef __CUDA
        input.to_gpu();
        weight.to_gpu();
        bias.to_gpu();
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
    layers::fc_backward(
        input, weight, bias, output, input_grad, weight_grad, bias_grad, output_grad);

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

TEST_F(TestConv2dLayer, TestConv2dForward) {
    layers::conv2d_forward(
        input,
        weight,
        bias,
        output,
        pad_h,
        pad_w,
        stride_h,
        stride_w
    );

#ifdef __CUDA
    output.to_cpu();
#endif

    for (int i = 0; i < batch_size * out_channels * height * width; ++i) {
        EXPECT_NEAR(output.get_data()[i], y[i], 1e-4);
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