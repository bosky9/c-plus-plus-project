cmake_minimum_required(VERSION 3.20)
project(c_plus_plus_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# INCLUDES
include_directories(${PROJECT_SOURCE_DIR}/include)

# MAIN
file(GLOB_RECURSE FAMILIES ${PROJECT_SOURCE_DIR}/src/families/*.cpp)
add_executable(main)
target_sources(main PRIVATE ${FAMILIES} src/main.cpp)

# TESTS
add_subdirectory(lib/Catch2)
add_executable(tests)
file(GLOB_RECURSE TESTS_LIST ${PROJECT_SOURCE_DIR}/test/*.cpp)
target_sources(tests PRIVATE ${FAMILIES} ${TESTS_LIST})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)