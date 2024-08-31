#include "imageExtractor.hpp"


int datasetExtractor(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << "Error: The specified path does not exist." << std::endl;
        return -1;
    }

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        std::string pathName = entry.path().string();
        std::string fileName = entry.path().filename().string();

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".csv")
        {
            size_t lastIndex = fileName.find_last_of(".");
            std::string rawName = fileName.substr(0, lastIndex);

            // Get the parent path
            lastIndex = path.find_last_of("/");
            std::string prevPath = path.substr(0, lastIndex);

            std::string outputPath = makeFolder(prevPath, rawName);

            int totalImagesConverted = importCSVDataset(pathName, outputPath);

            if (totalImagesConverted < 0)
            {
                std::cerr << "An error Occurred when try to build the images from the csv." << pathName << std::endl;
                return -1;

            } else if (totalImagesConverted == 0)
            {
                std::cerr << "The output directory " << outputPath << " is not empty. The dataset " << rawName << " already exist." << std::endl;

            } else
            {
                std::cout << "Successfully converted " << totalImagesConverted << " images from " << fileName << std::endl;
            }
        }
    }

    return 0;
}


int importCSVDataset(const std::string& csvFilePath, const std::string& outputDir)
{
    if (!std::filesystem::is_empty(outputDir))
    {
        return 0;
    }

    std::cout << "Importing dataset from: " << csvFilePath << std::endl;
    std::ifstream file(csvFilePath);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open the file " << csvFilePath << std::endl;
        return -1;
    }

    std::string line;
    std::vector<std::vector<int>> dataset;

    // Skip the first line
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<int> pixels;

        while (std::getline(ss, token, ','))
        {
            try {
                // Convert token to int
                pixels.push_back(std::stoi(token));
            }
            catch (const std::invalid_argument&) {
                std::cerr << "Warning: Invalid token '" << token << "' in line: " << line << std::endl;
            }
            catch (const std::out_of_range&) {
                std::cerr << "Warning: Token '" << token << "' is out of range in line: " << line << std::endl;
            }
        }

        // Ensure there are enough pixels for an MNIST image (1 label + 784 pixels)
        if (pixels.size() < 785) {
            std::cerr << "Error: Not enough pixel values in line: " << line << std::endl;
            continue;
        }

        // Add the read pixels to the dataset
        dataset.push_back(pixels);
    }

    file.close();

    int totalImagesConverted = csvToImages(dataset, outputDir);

    return totalImagesConverted; 
}


int csvToImages(const std::vector<std::vector<int>>& dataset, const std::string& outputDir) 
{
    int imageCounter = 0;

    // Process the dataset to create images
    for (const auto& pixels : dataset)
    {
        // The first value is the label
        int label = pixels[0];
        
        // The rest are pixel values (784 values for MNIST)
        std::vector<int> pixelValues(pixels.begin() + 1, pixels.end());

        // Create a 28x28 image from the pixel values
        cv::Mat img(28, 28, CV_8UC1);
        for (int i = 0; i < 28 * 28; ++i)
        {
            img.at<uchar>(i / 28, i % 28) = static_cast<uchar>(pixelValues[i]);
        }

        std::stringstream filename;
        filename << outputDir << "/image_" << label << "_" << imageCounter++ << ".png";
        // std::cout << "Saving image: " << filename.str() << std::endl;

        if (!cv::imwrite(filename.str(), img)) {
            std::cerr << "Error: Could not save image: " << filename.str() << std::endl;
            return -1;
        }
    }

    if (imageCounter == 0) {
        std::cerr << "Error: No images were created." << std::endl;
        return -1;
    }

    return imageCounter;
}