cmake_minimum_required(VERSION 3.20)
project(c_plus_plus_project)

# Set basic options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(G++_FLAGS "-g --coverage -pg")
set(CMAKE_CXX_FLAGS "${G++_FLAGS} -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror -fsanitize=undefined -fno-omit-frame-pointer -DWITHOUT_NUMPY")
set(DCMAKE_BUILD_TYPE "Debug")

# Set Valgrind
set(CTEST_MEMCHECK_TEST TRUE)
find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
INCLUDE(Dart)
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

# Fetch sciplot library for plots
FetchContent_Declare(sciplot_content
        GIT_REPOSITORY https://github.com/sciplot/sciplot.git
        GIT_TAG master)
FetchContent_GetProperties(sciplot_content)
if (NOT sciplot_content_POPULATED)
    FetchContent_Populate(sciplot_content)
endif ()
include_directories(${sciplot_content_SOURCE_DIR})

# Perform tests
add_executable(tests)
file(GLOB_RECURSE SRCS_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE TESTS_LIST ${PROJECT_SOURCE_DIR}/test/*.cpp)
list(REMOVE_ITEM TESTS_LIST ${PROJECT_SOURCE_DIR}/test/arima/test_performance.cpp)
target_sources(tests PRIVATE ${SRCS_LIST} ${TESTS_LIST})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# Add performances test
add_executable(test_performance ${PROJECT_SOURCE_DIR}/test/arima/test_performance.cpp)
target_sources(test_performance PRIVATE ${SRCS_LIST})
target_link_libraries(test_performance PRIVATE Catch2::Catch2WithMain)

# Add test exclusively for ARIMA (to check with Valgrind)
add_executable(test_arima ${PROJECT_SOURCE_DIR}/test/arima/test_arima.cpp)
target_sources(test_arima PRIVATE ${SRCS_LIST})
target_link_libraries(test_arima PRIVATE Catch2::Catch2WithMain)

include(CTest)
add_test(AllTests tests)
add_test(TestPerformance test_performance)
add_test(TestArima test_arima)

# Generate documentation with Doxygen with "make doc"
find_package(Doxygen)
if (Doxygen_FOUND)
    add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
            COMMENT "Generating Doxygen documentation")
endif ()

# Move library in bin
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src DESTINATION ${PROJECT_SOURCE_DIR}/bin)