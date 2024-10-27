add_executable(test_layers test_layers.cpp)
target_include_directories(test_layers PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(test_layers layers GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_layers)