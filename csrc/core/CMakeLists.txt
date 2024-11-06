add_subdirectory(device)
add_subdirectory(memory)
add_subdirectory(kernels)

add_library(core STATIC core.cpp)
target_link_libraries(core device kernels memory)