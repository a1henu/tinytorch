set(TENSOR_SOURCES
    tensor.cpp
)

add_library(tensor ${TENSOR_SOURCES})
target_include_directories(tensor PUBLIC ${CMAKE_SOURCE_DIR}/csrc/core/memory)
target_link_libraries(tensor)