add_executable(test_memory test_memory_cpu.cpp)

target_include_directories(test_memory PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_memory memory GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_memory)