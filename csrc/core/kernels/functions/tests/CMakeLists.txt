set(TEST_FUNCTIONS_SOURCES)

if(CUDA)
    list(APPEND TEST_FUNCTIONS_SOURCES test_functions_gpu.cu)
else()
    list(APPEND TEST_FUNCTIONS_SOURCES test_functions_cpu.cpp)
endif()

add_executable(test_functions ${TEST_FUNCTIONS_SOURCES})

target_include_directories(test_functions PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_functions kernels GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_functions)