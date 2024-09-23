cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(TEST "Build all tests." OFF)

if(TEST)
    add_subdirectory(third_party/googletest)

    include(CTest)
    include(GoogleTest)
    enable_testing()
endif()

include_directories(${PROJECT_SOURCE_DIR}/csrc)

add_subdirectory(csrc)

# add_subdirectory(third_party/pybind11)