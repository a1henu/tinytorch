set(MEMORY_SOURCES)

list(APPEND MEMORY_SOURCES memory.cpp)

if(CUDA)
    list(APPEND MEMORY_SOURCES memory.cu)
endif()

add_library(memory STATIC ${MEMORY_SOURCES})
target_include_directories(memory PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

if(TEST)
    add_subdirectory(tests)
endif()