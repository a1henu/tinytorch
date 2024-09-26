add_executable(test_tensor_at test_tensor_at.cpp)

target_include_directories(test_tensor_at PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_tensor_at tensor GTest::gtest_main)

add_executable(test_tensor_ops test_tensor_ops.cpp)
target_include_directories(test_tensor_ops PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_tensor_ops tensor GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor_at)
gtest_discover_tests(test_tensor_ops)