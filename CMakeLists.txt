cmake_minimum_required(VERSION 3.20)
project(c_plus_plus_project)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(c_plus_plus_project
        src/families/family.cpp
        src/families/normal.cpp
        src/tests.cpp)
