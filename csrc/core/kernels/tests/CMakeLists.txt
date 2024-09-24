set(TEST_OPS_SOURCES)

if(CUDA)
    list(APPEND TEST_OPS_SOURCES test_ops_gpu.cu)
else()
    list(APPEND TEST_OPS_SOURCES test_ops_cpu.cpp)
endif()

add_executable(test_ops ${TEST_OPS_SOURCES})

target_include_directories(test_ops PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_ops kernels GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_ops)