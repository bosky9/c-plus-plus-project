# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2021.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2021.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ettorec/CLionProjects/c-plus-plus-project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/c_plus_plus_project.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/c_plus_plus_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c_plus_plus_project.dir/flags.make

CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.o: CMakeFiles/c_plus_plus_project.dir/flags.make
CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.o: ../src/families/family.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.o -c /home/ettorec/CLionProjects/c-plus-plus-project/src/families/family.cpp

CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ettorec/CLionProjects/c-plus-plus-project/src/families/family.cpp > CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.i

CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ettorec/CLionProjects/c-plus-plus-project/src/families/family.cpp -o CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.s

CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.o: CMakeFiles/c_plus_plus_project.dir/flags.make
CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.o: ../src/families/normal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.o -c /home/ettorec/CLionProjects/c-plus-plus-project/src/families/normal.cpp

CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ettorec/CLionProjects/c-plus-plus-project/src/families/normal.cpp > CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.i

CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ettorec/CLionProjects/c-plus-plus-project/src/families/normal.cpp -o CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.s

CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.o: CMakeFiles/c_plus_plus_project.dir/flags.make
CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.o: ../src/tests.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.o -c /home/ettorec/CLionProjects/c-plus-plus-project/src/tests.cpp

CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ettorec/CLionProjects/c-plus-plus-project/src/tests.cpp > CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.i

CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ettorec/CLionProjects/c-plus-plus-project/src/tests.cpp -o CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.s

CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.o: CMakeFiles/c_plus_plus_project.dir/flags.make
CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.o: ../src/families/flat.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.o -c /home/ettorec/CLionProjects/c-plus-plus-project/src/families/flat.cpp

CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ettorec/CLionProjects/c-plus-plus-project/src/families/flat.cpp > CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.i

CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ettorec/CLionProjects/c-plus-plus-project/src/families/flat.cpp -o CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.s

# Object files for target c_plus_plus_project
c_plus_plus_project_OBJECTS = \
"CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.o" \
"CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.o" \
"CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.o" \
"CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.o"

# External object files for target c_plus_plus_project
c_plus_plus_project_EXTERNAL_OBJECTS =

c_plus_plus_project: CMakeFiles/c_plus_plus_project.dir/src/families/family.cpp.o
c_plus_plus_project: CMakeFiles/c_plus_plus_project.dir/src/families/normal.cpp.o
c_plus_plus_project: CMakeFiles/c_plus_plus_project.dir/src/tests.cpp.o
c_plus_plus_project: CMakeFiles/c_plus_plus_project.dir/src/families/flat.cpp.o
c_plus_plus_project: CMakeFiles/c_plus_plus_project.dir/build.make
c_plus_plus_project: CMakeFiles/c_plus_plus_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable c_plus_plus_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c_plus_plus_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/c_plus_plus_project.dir/build: c_plus_plus_project
.PHONY : CMakeFiles/c_plus_plus_project.dir/build

CMakeFiles/c_plus_plus_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c_plus_plus_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c_plus_plus_project.dir/clean

CMakeFiles/c_plus_plus_project.dir/depend:
	cd /home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ettorec/CLionProjects/c-plus-plus-project /home/ettorec/CLionProjects/c-plus-plus-project /home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug /home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug /home/ettorec/CLionProjects/c-plus-plus-project/cmake-build-debug/CMakeFiles/c_plus_plus_project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/c_plus_plus_project.dir/depend

