#include "tester.hpp"


std::vector<std::string> getJsonFiles(const std::string& folderPath)
{
    std::vector<std::string> jsonFiles;

    // Check if the directory exists
    if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath))
    {
        std::cerr << "Error: Given path is not a directory or does not exist." << std::endl;
        return jsonFiles;
    }

    // Iterate through the directory
    for (const auto& entry : std::filesystem::directory_iterator(folderPath))
    {
        // Check if the entry is a regular file and has a .json extension
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            jsonFiles.push_back(entry.path().string());
    }

    return jsonFiles;
}


void removeJsonFiles(const std::vector<std::string>& jsonFiles)
{
    for (const auto& file : jsonFiles)
    {
        try
        {
            std::filesystem::remove(file);
            std::cout << "Removed: " << file << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e)
        {
            std::cerr << "Error removing file " << file << ": " << e.what() << std::endl;
        }
    }
}