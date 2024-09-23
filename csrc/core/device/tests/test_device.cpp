#include <gtest/gtest.h>
#include <iostream>

#include "core/device/device.h"

class TestDevice : public ::testing::Test {
protected:
    device::CPU* cpu_device;
    device::GPU* gpu_device;

    void SetUp() override {
        cpu_device = new device::CPU();
        gpu_device = new device::GPU();
    }

    void TearDown() override {
        delete cpu_device;
        delete gpu_device;
    }
};

TEST_F(TestDevice, CPUIsCPU) {
    EXPECT_TRUE(cpu_device->is_cpu());
    EXPECT_FALSE(cpu_device->is_gpu());
}

TEST_F(TestDevice, GPUIsGPU) {
    EXPECT_TRUE(gpu_device->is_gpu());
    EXPECT_FALSE(gpu_device->is_cpu());
}

int main(int argc, char **argv) {
    std::cout << "run test for CORE::DEVICE" << std::endl << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}