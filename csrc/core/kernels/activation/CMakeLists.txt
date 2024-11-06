cmake_policy(SET CMP0076 NEW)

list(APPEND ACTIVATION_SOURCES
    relu.cpp
    sigmoid.cpp
    softmax.cpp
)

if(CUDA)
    list(APPEND ACTIVATION_SOURCES
        relu.cu
        sigmoid.cu
        softmax.cu
    )
endif()

target_sources(kernels PRIVATE ${ACTIVATION_SOURCES})

if(TEST)
    add_subdirectory(tests)
endif()