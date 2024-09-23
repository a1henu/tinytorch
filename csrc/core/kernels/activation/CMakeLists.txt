set(ACTIVATION_SOURCES)

list(APPEND ACTIVATION_SOURCES
    relu.cpp
    sigmoid.cpp
)

if(CUDA)
    list(APPEND ACTIVATION_SOURCES
        relu.cu
        sigmoid.cu
    )
endif()

add_library(activation STATIC ${ACTIVATION_SOURCES})
target_include_directories(activation PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

if(TEST)
    add_subdirectory(tests)
endif()