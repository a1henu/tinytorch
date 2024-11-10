cmake_minimum_required(VERSION 3.15)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
)

option(TEST "Build all tests." OFF)
option(CUDA "Build with CUDA support." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory(third_party/pybind11)

find_package(BLAS REQUIRED)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# set package name to tinytorch
set(TARGET_PACK tinytorch)
set(CMAKE_INSTALL_RPATH "${PYTHON_SITE_PACKAGES}/${TARGET_PACK}")

if(CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    add_compile_definitions(__CUDA)
endif()

if(TEST)
    add_subdirectory(third_party/googletest)

    include(CTest)
    include(GoogleTest)
    enable_testing()
endif()

include_directories(${PROJECT_SOURCE_DIR}/csrc)

add_subdirectory(csrc)
