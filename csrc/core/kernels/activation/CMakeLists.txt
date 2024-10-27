cmake_policy(SET CMP0076 NEW)

list(APPEND ACTIVATION_SOURCES
    relu.cpp
    sigmoid.cpp
)

if(CUDA)
    add_compile_definitions(RELU__IMP__CUDA)
    list(APPEND ACTIVATION_SOURCES
        relu.cu
        sigmoid.cu
    )
endif()

target_sources(kernels PRIVATE ${ACTIVATION_SOURCES})

if(TEST)
    add_subdirectory(tests)
endif()