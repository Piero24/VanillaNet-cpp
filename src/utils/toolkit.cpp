#include "toolkit.hpp"


std::string makeFolder(const std::string& basePath, const std::string& folderName) {
    std::string outputPath = basePath + "/" + folderName;

    if (!std::filesystem::exists(outputPath)) {
        std::filesystem::create_directories(outputPath);
    }

    return outputPath;
}