add_executable(test_device test_device.cpp)

target_include_directories(test_device PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_device device GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_device)