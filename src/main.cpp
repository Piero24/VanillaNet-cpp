
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
// *** clear && ./cpp-mnist
// *** 
// ***                                             ALL COMANDS IN TWO LINES
// ***
// *** MACOS
// *** rm -r build && mkdir build && cmake -S . -B build
// *** make -C build && clear && ./cpp-mnist
// ***
// ********************************************************************************************************************* 
// *********************************************************************************************************************

#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}