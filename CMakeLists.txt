cmake_minimum_required(VERSION 3.20)
project(c_plus_plus_project)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

file(GLOB_RECURSE FAMILIES ${PROJECT_SOURCE_DIR}/src/families/*.cpp)
add_executable(c_plus_plus_project ${FAMILIES} src/tests.cpp)
