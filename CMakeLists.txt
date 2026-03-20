cmake_minimum_required(VERSION 3.10)
project(Nemu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Core library: only utility code (spill.c)
add_library(core
    src/utils/spill.c
    src/utils/spill.h
)

# Include path for core and all targets
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Serializer executable
add_executable(serializer src/serialize.c)
target_compile_definitions(serializer PRIVATE SERIALIZER)
target_link_libraries(serializer PRIVATE core z m)

# Loader builtin test
add_executable(loader_test
    src/loader.c
    src/serialize.c
)
target_compile_definitions(loader_test PRIVATE LOADER_TEST_MAIN)
target_link_libraries(loader_test PRIVATE core z m)

# Visualizer test
add_executable(visualizer
    tests/visualizer.c
    src/loader.c
)
target_link_libraries(visualizer PRIVATE core z m)