cmake_minimum_required(VERSION 3.20)
project(c_plus_plus_project)

# Set basic options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(G++_FLAGS "-g --coverage")
set(CMAKE_CXX_FLAGS "${G++_FLAGS} -O3 -Wall -Wextra -Wpedantic -Werror -fsanitize=undefined -fno-omit-frame-pointer")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(DCMAKE_BUILD_TYPE "Debug")

# Set Valgrind
set(CTEST_MEMCHECK_TEST TRUE)
find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
include(Dart)
set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --track-fds=yes --track-origins=yes")

# Fetch Eigen library
#find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Fetch Python (required by matplotlibcpp.hpp)
find_package(Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# Fetch Catch2 library for tests
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.0.0-preview3
)
FetchContent_MakeAvailable(Catch2)

# Include headers and external libraries
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party)

# Perform tests
add_executable(tests)
file(GLOB_RECURSE SRCS_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE TESTS_LIST ${PROJECT_SOURCE_DIR}/test/*.cpp)
target_sources(tests PRIVATE ${SRCS_LIST} ${TESTS_LIST})
target_link_libraries(tests PRIVATE Python3::Python Catch2::Catch2WithMain)

include(CTest)
add_test(NAME AllTests
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
        COMMAND tests)

# Generate documentation with Doxygen with "make doc"
find_package(Doxygen)
if (Doxygen_FOUND)
    add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
            COMMENT "Generating Doxygen documentation")
endif ()

# Move executables in bin
#add_custom_target(install)