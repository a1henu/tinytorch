set(KERNELS_SOURCES)

list(APPEND KERNELS_SOURCES ops.cpp)

if(CUDA)
    list(APPEND KERNELS_SOURCES ops.cu)
    find_package(CUDA REQUIRED)
endif()

add_library(kernels STATIC ${KERNELS_SOURCES})
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

if(CUDA)
    target_link_libraries(kernels PUBLIC ${BLAS_LIBRARIES} ${CUDA_LIBRARIES} cublas)
else()
    target_link_libraries(kernels PUBLIC ${BLAS_LIBRARIES})
endif()

add_subdirectory(functions)

if(TEST)
    add_subdirectory(tests)
endif()