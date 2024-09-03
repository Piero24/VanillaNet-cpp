#include "toolkit.hpp"


std::string makeFolder(const std::string& basePath, const std::string& folderName) {
    std::string outputPath = basePath + "/" + folderName;

    if (!std::filesystem::exists(outputPath)) {
        std::filesystem::create_directories(outputPath);
    }

    return outputPath;
}


int labelExtractor(const std::string& imagePath) {
    std::string label = imagePath.substr(imagePath.find_last_of("_") - 1, 1);
    return std::stoi(label);
}


std::vector<double> trueLabel(int label) {
    std::vector<double> labelVector(10, 0.0);
    labelVector[label] = 1.0;

    return labelVector;
}