set(LAYERS_SOURCES
    fc_layer.cpp
    pooling_layer.cpp
    conv2d_layer.cpp
    softmax_layer.cpp
    cross_entropy_layer.cpp
)

set(LAYERS_BINDING_SOURCES
    funcs_binding.cpp
)

add_library(layers STATIC ${LAYERS_SOURCES})
target_include_directories(layers PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(layers tensor core)

pybind11_add_module(_libfuncs ${LAYERS_BINDING_SOURCES})
target_link_libraries(_libfuncs PRIVATE pybind11::headers layers)
target_compile_definitions(_libfuncs PRIVATE VERSION_INFO=${PROJECT_VERSION})

set_target_properties(_libfuncs PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS _libfuncs DESTINATION ${TARGET_PACK})

if(TEST)
    add_subdirectory(tests)
endif()