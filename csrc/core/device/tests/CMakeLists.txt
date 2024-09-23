if(TEST)
    enable_testing()
    add_executable(device_test device_test.cpp)

    target_include_directories(device_test PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
    target_link_libraries(device_test device GTest::gtest_main)

    include(GoogleTest)
    #gtest_discover_tests(device_test)
    add_test(DEVICE_TEST device_test ${CMAKE_BINARY_DIR}/csrc/core/device/tests/device_test)
endif()