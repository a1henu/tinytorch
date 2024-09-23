set(MEMORY_SOURCES
    memory.cpp
)

# TODO: Add cuda support
# maybe we need to add a flag to enable cuda support

add_library(memory STATIC ${MEMORY_SOURCES})
target_include_directories(memory PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

if(TEST)
    add_subdirectory(tests)
endif()