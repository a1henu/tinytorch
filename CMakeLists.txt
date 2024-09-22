cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_subdirectory(csrc/tensor)
add_subdirectory(csrc/core)

# add_subdirectory(third_party/pybind11)

include_directories(${PROJECT_SOURCE_DIR}/csrc)

if(TEST)
    add_subdirectory(third_party/googletest)
endif()