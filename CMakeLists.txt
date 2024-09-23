cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

option(TEST "Build all tests." OFF)
option(CUDA "Build with CUDA support." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

if(TEST)
    add_subdirectory(third_party/googletest)

    include(CTest)
    include(GoogleTest)
    enable_testing()
endif()

include_directories(${PROJECT_SOURCE_DIR}/csrc)

add_subdirectory(csrc)

# add_subdirectory(third_party/pybind11)