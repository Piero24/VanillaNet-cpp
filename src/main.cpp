
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
// *** clear && ./VanillaNet-cpp
// *** 
// ***                                             ALL COMANDS IN TWO LINES
// ***
// *** MACOS
// *** rm -r build && mkdir build && cmake -S . -B build
// *** make -C build && clear && ./VanillaNet-cpp
// ***
// ********************************************************************************************************************* 
// *********************************************************************************************************************

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "imageExtractor.hpp"
#include "neuron.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "toolkit.hpp"
#include "lossFunctions.hpp"


int main() {
    
    // datasetExtractor("./Resources/Dataset/csv");
    std::string imagePath = "./Resources/Dataset/mnist_train/image_0_1.png";

    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    cv::Mat fImage; 
    inputImage.convertTo(fImage, CV_64F);
    std::vector<double> imagePixelVector(fImage.begin<double>(), fImage.end<double>());
    // printf("Image size: %ld\n", imagePixelVector.size());

    Layer hiddenLayer(10, imagePixelVector);
    std::vector<double> outputHidden = Activation(ActivationType::RELU, hiddenLayer.outputs);

    Layer outputLayer(10, outputHidden);
    std::vector<double> outputOput = Activation(ActivationType::SOFTMAX, outputLayer.outputs);

    int label = labelExtractor(imagePath);
    std::vector<double> labelVector = trueLabel(label);


    double totalSum = 0.0;
    int maxIndex = 0;
    double maxVal = 0.0;

    for (int i = 0; i < outputOput.size(); i++)
    {
        printf("Output %d: %f\n", i, outputOput[i]);
        totalSum += outputOput[i];
        if (outputOput[i] > maxVal)
        {
            maxVal = outputOput[i];
            maxIndex = i;
        }
    }

    printf("Output size: %ld - Total sum: %f\n", outputOput.size(), totalSum);
    double loss = mse_loss(labelVector, outputOput);

    printf("True value: %d, Predicted Value: %d with probability: %f, Loss: %f\n", label, maxIndex, maxVal, loss);

    return 0;
}