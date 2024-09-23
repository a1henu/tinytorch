cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/csrc)

# add_subdirectory(csrc/tensor)
add_subdirectory(csrc/core)

# add_subdirectory(third_party/pybind11)

if(TEST)
    add_subdirectory(third_party/googletest)

    include(CTest)
    include(GoogleTest)
    enable_testing()
endif()