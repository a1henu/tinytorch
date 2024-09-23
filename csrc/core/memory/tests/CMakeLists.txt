set(TEST_MEMORY_SOURCES)

if(CUDA)
    list(APPEND TEST_MEMORY_SOURCES test_memory_gpu.cu)
else()
    list(APPEND TEST_MEMORY_SOURCES test_memory_cpu.cpp)
endif()

add_executable(test_memory ${TEST_MEMORY_SOURCES})

target_include_directories(test_memory PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_memory memory GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_memory)