set(TEST_TENSOR_OPS_SOURCES)

list(APPEND 
    TEST_TENSOR_OPS_SOURCES 
    test_tensor_at.cpp
)

add_executable(test_tensor_ops ${TEST_TENSOR_OPS_SOURCES})

target_include_directories(test_tensor_ops PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_tensor_ops tensor_ops GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor_ops)