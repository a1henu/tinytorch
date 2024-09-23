set(ACTIVATION_SOURCES
    relu.cpp
    sigmoid.cpp
)

add_library(activation STATIC ${ACTIVATION_SOURCES})
target_include_directories(activation PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

if(TEST)
    add_subdirectory(tests)
endif()