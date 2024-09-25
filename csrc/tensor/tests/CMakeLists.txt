set(TEST_TENSOR_SOURCES)

list(APPEND 
    TEST_TENSOR_SOURCES 
    test_tensor.cpp
)

add_executable(test_tensor ${TEST_TENSOR_SOURCES})

target_include_directories(test_tensor PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_tensor tensor GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor)