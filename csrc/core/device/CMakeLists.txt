set(DEVICE_SOURCES
    device.cpp
)

add_library(device STATIC ${DEVICE_SOURCES})
target_include_directories(device PUBLIC ${CMAKE_SOURCE_DIR}/csrc)

if(TEST)
    add_subdirectory(tests)
endif()