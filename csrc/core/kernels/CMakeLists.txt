set(KERNELS_SOURCES)

list(APPEND KERNELS_SOURCES ops.cpp)

if(CUDA)
    list(APPEND KERNELS_SOURCES ops.cu)
endif()

add_library(kernels STATIC ${KERNELS_SOURCES})
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

add_subdirectory(activation)

if(TEST)
    add_subdirectory(tests)
endif()