
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


int main() {
    
    // datasetExtractor("./Resources/Dataset/csv");

    cv::Mat inputImage = cv::imread("./Resources/Dataset/mnist_train/image_0_1.png", cv::IMREAD_GRAYSCALE);

    cv::Mat fImage; 
    inputImage.convertTo(fImage, CV_64F);
    std::vector<double> imagePixelVector(fImage.begin<double>(), fImage.end<double>());
    // printf("Image size: %ld\n", imagePixelVector.size());

    Layer hiddenLayer(10, imagePixelVector);
    std::vector<double> outputHidden = Activation(ActivationType::RELU, hiddenLayer.outputs);

    Layer outputLayer(10, outputHidden);
    std::vector<double> outputOput = Activation(ActivationType::SOFTMAX, outputLayer.outputs);


    double totalSum = 0.0;

    for (int i = 0; i < outputOput.size(); i++)
    {
        printf("Output %d: %f\n", i, outputOput[i]);
        totalSum += outputOput[i];
    }

    printf("Output size: %ld - Total sum: %f\n", outputOput.size(), totalSum);

    return 0;
}