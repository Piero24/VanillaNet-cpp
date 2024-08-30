cmake_minimum_required(VERSION 3.20)

# Project name
project(cpp-mnist)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add an executable
add_executable(cpp-mnist src/main.cpp)