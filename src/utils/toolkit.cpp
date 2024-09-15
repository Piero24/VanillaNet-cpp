#include "toolkit.hpp"


std::string makeFolder(const std::string& basePath, const std::string& folderName)
{
    std::string outputPath = basePath + "/" + folderName;

    if (!std::filesystem::exists(outputPath))
    {
        std::filesystem::create_directories(outputPath);
    }

    return outputPath;
}


int labelExtractor(const std::string& imagePath)
{
    std::string label = imagePath.substr(imagePath.find_last_of("_") - 1, 1);
    return std::stoi(label);
}


std::vector<double> trueLabel(int label)
{
    std::vector<double> labelVector(10, 0.0);
    labelVector[label] = 1.0;

    return labelVector;
}


std::vector<std::string> datasetImagesVector(const std::string& datasetPath)
{
    size_t idx = datasetPath.find_last_of(".");
    std::string format = datasetPath.substr(idx, datasetPath.size());
    if (format == ".png" || format == ".jpg" || format == ".jpeg")
    {
        return {datasetPath};
    }

    std::vector<std::string> datasetImages;
    for (const auto& entry : std::filesystem::directory_iterator(datasetPath))
    {
        datasetImages.push_back(entry.path());
    }

    return datasetImages;
}


int parser(Arguments& inputParams, int argc, char** inputToParse)
{
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(inputToParse[i], "-Train") == 0 || strcmp(inputToParse[i], "-Tr") == 0)
        {
            inputParams.Train = true;
            inputParams.TrainDatasetPath = inputToParse[i + 1];
            inputParams.TrainDatasetImages = datasetImagesVector(inputParams.TrainDatasetPath);
        }
        else if (strcmp(inputToParse[i], "-Test") == 0 || strcmp(inputToParse[i], "-Te") == 0)
        {
            inputParams.Test = true;
            inputParams.TestDatasetPath = inputToParse[i + 1];
            inputParams.TestDatasetImages = datasetImagesVector(inputParams.TestDatasetPath);
        } 
        else if (strcmp(inputToParse[i], "-csv") == 0)
        {
            if (inputParams.Train || inputParams.Test)
            {
                std::cout << "One operation at time. Now extract the datasets. At the next call you can use -Train and/or -Test." << std::endl;
            }
            printf("Dataset path: %s\n", inputToParse[i + 1]);
            // datasetExtractor("./Resources/Dataset/csv");
            datasetExtractor(inputToParse[i + 1]);
            return 1;
        }
        else if (strcmp(inputToParse[i], "-WeightsBiases") == 0 || strcmp(inputToParse[i], "-wb") == 0)
        {
            inputParams.WeightsBiasesPath = inputToParse[i + 1];
            inputParams.hasWeightsBiases = true;
        }
        else if (strcmp(inputToParse[i], "-Epochs") == 0 || strcmp(inputToParse[i], "-E") == 0)
        {
            inputParams.epochs = std::stoi(inputToParse[i + 1]);
        }
        else if (strcmp(inputToParse[i], "-LearningRate") == 0 || strcmp(inputToParse[i], "-LR") == 0)
        {
            inputParams.learningRate = std::stod(inputToParse[i + 1]);
        }
        else if (strcmp(inputToParse[i], "-BatchSize") == 0 || strcmp(inputToParse[i], "-BS") == 0)
        {
            inputParams.batchSize = std::stoi(inputToParse[i + 1]);
        }
        else if (strcmp(inputToParse[i], "-print") == 0 || strcmp(inputToParse[i], "-p") == 0)
        {
            inputParams.print = true;
        } 
        else if (strcmp(inputToParse[i], "-help") == 0 || strcmp(inputToParse[i], "-h") == 0)
        {
            // TODO
            std::cout << "Check here for the avable parameters: https://github.com/Piero24/VanillaNet-cpp/blob/main/.github/doc.md" << std::endl;
            return -1;
        } 
        // else
        // {
        //     std::cout << "Invalid input parameter. Use -help for info on the required params." << std::endl;
        //     return -1;
        // }
    }

    if (!inputParams.Train && !inputParams.Test)
    {
        std::cout << "Please select a mode: -Train or -Test. Or use -csv for extract the datasets." << std::endl;
        return -1;
    }

    if (inputParams.Train && strcmp(inputParams.TrainDatasetPath.c_str(), "") == 0)
    {
        std::cout << "Training mode selected. Please provide a training dataset path." << std::endl;
        return -1;
    }
    if (inputParams.Train && (inputParams.epochs == 0 || inputParams.learningRate == 0.0 || inputParams.batchSize == 0))
    {
        std::cout << "Training mode selected. Please provide the number of epochs, learning rate, and batch size." << std::endl;
        return -1;
    }
    if (inputParams.Test && strcmp(inputParams.TestDatasetPath.c_str(), "") == 0)
    {
        std::cout << "Testing mode selected. Please provide a testing dataset path." << std::endl;
        return -1;
    }

    return 0;
}


void imageToVectorAndLabel(VectorLabel& vecLabel, std::string imagePath)
{
    // std::string imagePath = "./Resources/Dataset/mnist_train/image_0_1.png";
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Convert the image to CV_64F (double precision) and normalize to [0, 1]
    cv::Mat fImage; 
    inputImage.convertTo(fImage, CV_64F, 1.0 / 255.0); // Normalize to [0, 1]
    
    // Create a vector from the image data
    std::vector<double> imagePixelVector(fImage.begin<double>(), fImage.end<double>());
    vecLabel.imagePixelVector = imagePixelVector;
    // printf("Image size: %ld\n", imagePixelVector.size());

    vecLabel.label = labelExtractor(imagePath);
    vecLabel.labelVector = trueLabel(vecLabel.label);
}


std::string getCurrentDateTime()
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // Create a tm structure to hold local time
    std::tm* localTime = std::localtime(&now_time);

    // Use a stringstream to format the date and time
    std::ostringstream oss;
    oss << std::put_time(localTime, "%m_%d_%y_%H_%M_%S");

    return oss.str();
}


std::string getCurrentDate()
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // Create a tm structure to hold local time
    std::tm* localTime = std::localtime(&now_time);

    // Use a stringstream to format the date and time
    std::ostringstream oss;
    oss << std::put_time(localTime, "%m_%d_%y");

    return oss.str();
}