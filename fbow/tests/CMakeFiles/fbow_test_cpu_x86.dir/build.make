# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ningjiang/metis/DBoW_testing/fbow/tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ningjiang/metis/DBoW_testing/fbow/tests

# Include any dependencies generated for this target.
include CMakeFiles/fbow_test_cpu_x86.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fbow_test_cpu_x86.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fbow_test_cpu_x86.dir/flags.make

CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.o: CMakeFiles/fbow_test_cpu_x86.dir/flags.make
CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.o: test_cpu_x86.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ningjiang/metis/DBoW_testing/fbow/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.o -c /home/ningjiang/metis/DBoW_testing/fbow/tests/test_cpu_x86.cpp

CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ningjiang/metis/DBoW_testing/fbow/tests/test_cpu_x86.cpp > CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.i

CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ningjiang/metis/DBoW_testing/fbow/tests/test_cpu_x86.cpp -o CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.s

# Object files for target fbow_test_cpu_x86
fbow_test_cpu_x86_OBJECTS = \
"CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.o"

# External object files for target fbow_test_cpu_x86
fbow_test_cpu_x86_EXTERNAL_OBJECTS =

fbow_test_cpu_x86: CMakeFiles/fbow_test_cpu_x86.dir/test_cpu_x86.o
fbow_test_cpu_x86: CMakeFiles/fbow_test_cpu_x86.dir/build.make
fbow_test_cpu_x86: CMakeFiles/fbow_test_cpu_x86.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ningjiang/metis/DBoW_testing/fbow/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fbow_test_cpu_x86"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fbow_test_cpu_x86.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fbow_test_cpu_x86.dir/build: fbow_test_cpu_x86

.PHONY : CMakeFiles/fbow_test_cpu_x86.dir/build

CMakeFiles/fbow_test_cpu_x86.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fbow_test_cpu_x86.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fbow_test_cpu_x86.dir/clean

CMakeFiles/fbow_test_cpu_x86.dir/depend:
	cd /home/ningjiang/metis/DBoW_testing/fbow/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ningjiang/metis/DBoW_testing/fbow/tests /home/ningjiang/metis/DBoW_testing/fbow/tests /home/ningjiang/metis/DBoW_testing/fbow/tests /home/ningjiang/metis/DBoW_testing/fbow/tests /home/ningjiang/metis/DBoW_testing/fbow/tests/CMakeFiles/fbow_test_cpu_x86.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fbow_test_cpu_x86.dir/depend

