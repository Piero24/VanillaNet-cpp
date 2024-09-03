#ifndef TOOLKIT_HPP
#define TOOLKIT_HPP

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>


/**
 * @brief Creates a directory if it does not already exist.
 * 
 * This function constructs a folder path by combining a base path and a folder 
 * name. If the specified directory does not exist, it creates the directory 
 * along with any necessary parent directories.
 * 
 * @param basePath The base path where the new folder will be created.
 * @param folderName The name of the folder to create.
 * 
 * @return The full path of the created or existing directory.
 */
std::string makeFolder(const std::string& basePath, const std::string& folderName);


/**
 * @brief 
 * 
 * @param imagePath 
 * 
 * @return
 */
int labelExtractor(const std::string& imagePath);


/**
 * @brief 
 * 
 * @param label 
 * 
 * @return
 */
std::vector<double> trueLabel(int label);


#endif // TOOLKIT_HPP