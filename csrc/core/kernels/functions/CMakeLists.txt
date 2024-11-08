cmake_policy(SET CMP0076 NEW)

list(APPEND FUNCTIONS_SOURCES
    relu.cpp
    sigmoid.cpp
    softmax.cpp
    cross_entropy.cpp
)

if(CUDA)
    list(APPEND FUNCTIONS_SOURCES
        relu.cu
        sigmoid.cu
        softmax.cu
    )
endif()

target_sources(kernels PRIVATE ${FUNCTIONS_SOURCES})

if(TEST)
    add_subdirectory(tests)
endif()