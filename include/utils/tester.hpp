#ifndef TESTER_HPP
#define TESTER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>


std::vector<std::string> getJsonFiles(const std::string& folderPath);


void removeJsonFiles(const std::vector<std::string>& jsonFiles);


#endif // TESTER_HPP