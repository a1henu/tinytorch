set(TENSOR_SOURCES
    tensor.cpp
)

add_subdirectory(operators)

add_library(tensor ${TENSOR_SOURCES})
target_include_directories(tensor PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(tensor tensor_ops)

if(TEST)
    add_subdirectory(tests)
endif()