#ifndef TESTER_HPP
#define TESTER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>


/**
 * @brief Retrieves all JSON file paths from a specified directory.
 * 
 * This function scans the given directory (`folderPath`) for regular files 
 * with a `.json` extension and collects their paths into a vector of strings. 
 * It provides a way to easily obtain a list of JSON files for further processing 
 * or analysis.
 * 
 * @param folderPath The path to the directory to be scanned for JSON files.
 * @return A vector of strings containing the paths of all JSON files found in 
 *         the specified directory. If the directory does not exist or is invalid, 
 *         an empty vector is returned.
 * 
 * @note If the given path is not a valid directory, an error message is printed 
 *       to the standard error stream, and an empty vector is returned.
 */
std::vector<std::string> getJsonFiles(const std::string& folderPath);


/**
 * @brief Removes specified JSON files from the filesystem.
 * 
 * This function takes a vector of JSON file paths and attempts to delete each 
 * file from the filesystem. It provides a way to clean up JSON files that 
 * are no longer needed or to manage file storage effectively.
 * 
 * @param jsonFiles A vector of strings containing the paths of the JSON files 
 *                  to be removed from the filesystem.
 * 
 * @note The function prints the name of each removed file. If an error occurs 
 *       during file removal (e.g., file not found, permission issues), an 
 *       error message is printed to the standard error stream.
 */
void removeJsonFiles(const std::vector<std::string>& jsonFiles);


#endif // TESTER_HPP