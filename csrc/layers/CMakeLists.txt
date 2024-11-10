set(LAYERS_SOURCES
    fc_layer.cpp
    pooling_layer.cpp
    conv2d_layer.cpp
    softmax_layer.cpp
    cross_entropy_layer.cpp
)

add_library(layers STATIC ${LAYERS_SOURCES})
target_include_directories(layers PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(layers tensor core)

if(TEST)
    add_subdirectory(tests)
endif()