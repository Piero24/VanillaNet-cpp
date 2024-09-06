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
        else if (strcmp(inputToParse[i], "-help") == 0 || strcmp(inputToParse[i], "-h") == 0)
        {
            // TODO
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

    cv::Mat fImage; 
    inputImage.convertTo(fImage, CV_64F);
    std::vector<double> imagePixelVector(fImage.begin<double>(), fImage.end<double>());
    vecLabel.imagePixelVector = imagePixelVector;
    // printf("Image size: %ld\n", imagePixelVector.size());

    vecLabel.label = labelExtractor(imagePath);
    vecLabel.labelVector = trueLabel(vecLabel.label);
}


void infoPrinter(Arguments& inputParams)
{
    if (inputParams.Train && inputParams.Test)
    {
        printf("Training and Testing mode selected.\n");
        printf("Training dataset path: %s\nTesting dataset path: %s\n", inputParams.TrainDatasetPath.c_str(), inputParams.TestDatasetPath.c_str());
    }
    else if (inputParams.Train)
    {
        printf("Training mode selected. Dataset path: %s\n", inputParams.TrainDatasetPath.c_str());
    }
    else if (inputParams.Test)
    {
        printf("Testing mode selected. Dataset path: %s\n", inputParams.TestDatasetPath.c_str());
    }
    else
    {
        printf("No mode selected.\n");
        exit(1);
    }
}