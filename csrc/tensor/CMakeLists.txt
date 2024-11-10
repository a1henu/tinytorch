add_subdirectory(operators)

set(TENSOR_SOURCES
    tensor.cpp
)

set(TENSOR_BINDING_SOURCES
    tensor_binding.cpp
)

add_library(tensor STATIC ${TENSOR_SOURCES})
target_include_directories(tensor PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(tensor tensor_ops)

pybind11_add_module(_libtensor ${TENSOR_BINDING_SOURCES})
target_link_libraries(_libtensor PRIVATE pybind11::headers tensor)
target_compile_definitions(_libtensor PRIVATE VERSION_INFO=${PROJECT_VERSION})

set_target_properties(_libtensor PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS _libtensor DESTINATION ${TARGET_PACK})

if(TEST)
    add_subdirectory(tests)
endif()