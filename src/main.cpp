
// ********************************************************************************************************************* 
// ********************************************************************************************************************* 
// ***
// ***                                                     COMMANDS
// ***
// *** MACOS
// *** # Remove the old build directory and create a new one
// *** rm -r build && mkdir build
// *** 
// *** # Generate build files in the build directory
// *** cmake -S . -B build
// *** 
// *** # Build the project inside the build directory
// *** make -C build
// *** 
// *** # Clear the terminal and run the executable from the main directory
// *** clear && ./VanillaNet-cpp
// *** 
// ***                                             ALL COMANDS IN TWO LINES
// ***
// *** MACOS
// *** rm -r build && mkdir build && cmake -S . -B build
// *** make -C build && clear && ./VanillaNet-cpp
// ***
// ********************************************************************************************************************* 
// *********************************************************************************************************************

#include <iostream>
#include <filesystem>
#include "imageExtractor.hpp"

int main() {
    
    datasetExtractor("./Resources/Dataset/csv");
    return 0;
}