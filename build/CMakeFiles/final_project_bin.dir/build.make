# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.10.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.10.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build

# Include any dependencies generated for this target.
include CMakeFiles/final_project_bin.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/final_project_bin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/final_project_bin.dir/flags.make

CMakeFiles/final_project_bin.dir/src/main.cpp.o: CMakeFiles/final_project_bin.dir/flags.make
CMakeFiles/final_project_bin.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/final_project_bin.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/final_project_bin.dir/src/main.cpp.o -c /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/src/main.cpp

CMakeFiles/final_project_bin.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/final_project_bin.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/src/main.cpp > CMakeFiles/final_project_bin.dir/src/main.cpp.i

CMakeFiles/final_project_bin.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/final_project_bin.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/src/main.cpp -o CMakeFiles/final_project_bin.dir/src/main.cpp.s

CMakeFiles/final_project_bin.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/final_project_bin.dir/src/main.cpp.o.requires

CMakeFiles/final_project_bin.dir/src/main.cpp.o.provides: CMakeFiles/final_project_bin.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/final_project_bin.dir/build.make CMakeFiles/final_project_bin.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/final_project_bin.dir/src/main.cpp.o.provides

CMakeFiles/final_project_bin.dir/src/main.cpp.o.provides.build: CMakeFiles/final_project_bin.dir/src/main.cpp.o


# Object files for target final_project_bin
final_project_bin_OBJECTS = \
"CMakeFiles/final_project_bin.dir/src/main.cpp.o"

# External object files for target final_project_bin
final_project_bin_EXTERNAL_OBJECTS =

final_project_bin: CMakeFiles/final_project_bin.dir/src/main.cpp.o
final_project_bin: CMakeFiles/final_project_bin.dir/build.make
final_project_bin: imgui/libimgui.a
final_project_bin: glfw/src/libglfw3.a
final_project_bin: glad/libglad.a
final_project_bin: CMakeFiles/final_project_bin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable final_project_bin"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/final_project_bin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/final_project_bin.dir/build: final_project_bin

.PHONY : CMakeFiles/final_project_bin.dir/build

CMakeFiles/final_project_bin.dir/requires: CMakeFiles/final_project_bin.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/final_project_bin.dir/requires

CMakeFiles/final_project_bin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/final_project_bin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/final_project_bin.dir/clean

CMakeFiles/final_project_bin.dir/depend:
	cd /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build /Users/bushite_macbook/Desktop/18Spring/GM/gm-final-project-Bushite/build/CMakeFiles/final_project_bin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/final_project_bin.dir/depend

