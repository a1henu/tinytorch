set(DEVICE_SOURCES
    device.cpp
)

add_library(device ${DEVICE_SOURCES})
target_include_directories(device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(device)

if(TEST)
    add_subdirectory(tests)
endif()