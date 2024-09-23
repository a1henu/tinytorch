set(TEST_ACTIVATION_SOURCES)

if(CUDA)
    list(APPEND TEST_ACTIVATION_SOURCES test_activation_gpu.cu)
else()
    list(APPEND TEST_ACTIVATION_SOURCES test_activation_cpu.cpp)
endif()

add_executable(test_activation ${TEST_ACTIVATION_SOURCES})

target_include_directories(test_activation PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_activation activation GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_activation)