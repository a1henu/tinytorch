set(LAYERS_SOURCES
    layers.cpp
)

add_library(layers STATIC ${LAYERS_SOURCES})
target_include_directories(layers PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(layers tensor)

if(TEST)
    add_subdirectory(tests)
endif()