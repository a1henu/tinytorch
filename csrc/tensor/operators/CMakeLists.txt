set(TENSOR_OPS_SOURCES)

list(APPEND TENSOR_OPS_SOURCES
    tensor_activation.cpp
)

add_library(tensor_ops STATIC ${TENSOR_OPS_SOURCES})
target_include_directories(tensor_ops PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(tensor_ops core)

if(TEST)
    add_subdirectory(tests)
endif()