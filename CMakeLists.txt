cmake_minimum_required(VERSION 3.20)

# Project name and version
project(VanillaNet-cpp VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set OpenCV directory if not found automatically
set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Boost Filesystem
find_package(Boost COMPONENTS filesystem REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Add the include directory for your project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectories for extractor and utils to include path
include_directories(${CMAKE_SOURCE_DIR}/include/extractor)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/include/network)
include_directories(${CMAKE_SOURCE_DIR}/include/nlohmann)

# Specify the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add executable
add_executable(VanillaNet-cpp 
    src/main.cpp
    src/extractor/imageExtractor.cpp
    src/utils/toolkit.cpp
    src/utils/tester.cpp
    src/utils/printer.cpp
    src/utils/saveToJson.cpp
    src/network/neuron.cpp
    src/network/activation.cpp
    src/network/layer.cpp
    src/network/network.cpp
    src/lossFunctions.cpp
    src/extractor/weightsBiasExtractor.cpp
    src/train.cpp
    src/test.cpp
    )

# Link libraries
target_link_libraries(VanillaNet-cpp ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Package settings (optional)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)