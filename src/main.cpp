
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

#include "network.hpp"
#include "toolkit.hpp"
#include "activation.hpp"
#include "lossFunctions.hpp"
#include "weightsBiasExtractor.hpp"
#include "train.hpp"
#include "test.hpp"
#include "printer.hpp"


int main(int argc, char **argv)
{
    Arguments inputParams;
    int res = parser(inputParams, argc, argv);
    if (res != 0) return res;

    Network net;
    net.addLayer(Layer(784, 128));
    net.addLayer(ActivationLayer(ActivationType::RELU));
    net.addLayer(Layer(128, 10));
    net.addLayer(ActivationLayer(ActivationType::SIGMOID));

    //! Remove after testing
    // inputParams.hasWeightsBiases = true;
    // inputParams.WeightsBiasesPath = "./Resources/output/weights/test.json";
    // inputParams.WeightsBiasesPath = "./Resources/output/weights/mnist_fc128_relu_fc10_log_softmax_weights_biases.json";
    inputParams.learningRate = 0.5;
    inputParams.batchSize = 60;
    inputParams.epochs = 4;
    //!
    
    infoPrinter(inputParams, net);

    std::vector<BiasesWeights> importedWeightsAndBiases;
    weightsBiasExtractor(inputParams, importedWeightsAndBiases);
    net.importWeightsBiases(importedWeightsAndBiases);

    // TRAIN
    networkTrain(net, inputParams);

    // TEST
    networkTest(net, inputParams);

    return 0;
}