cmake_minimum_required(VERSION 3.20)
project(c_plus_plus_project)

# Set basic options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(G++_FLAGS "-g --coverage")
set(CMAKE_CXX_FLAGS "${G++_FLAGS} -O3 -Wall -Wextra -Wpedantic -Werror -fsanitize=undefined -fno-omit-frame-pointer")
set(DCMAKE_BUILD_TYPE "Debug")

# Set Valgrind
set(CTEST_MEMCHECK_TEST TRUE)
find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
include(Dart)
set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --track-fds=yes --track-origins=yes")

# Include headers and external libraries
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party)

# Fetch Catch2 library for tests
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.0.0-preview3
)
FetchContent_MakeAvailable(Catch2)

# Fetch Python (required by matplotlibcpp.hpp)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Perform tests
add_executable(tests)
file(GLOB_RECURSE SRCS_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE TESTS_LIST ${PROJECT_SOURCE_DIR}/test/*.cpp)
target_sources(tests PRIVATE ${SRCS_LIST} ${TESTS_LIST})

target_link_libraries(tests PRIVATE Python3::Python Python3::Module Catch2::Catch2WithMain INTERFACE WITHOUT_NUMPY)

include(CTest)
add_test(NAME AllTests
        COMMAND tests)

# Generate documentation with Doxygen with "make doc"
find_package(Doxygen)
if (Doxygen_FOUND)
    add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
            COMMENT "Generating Doxygen documentation")
endif ()

# Move library in bin
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src DESTINATION ${PROJECT_SOURCE_DIR}/bin)