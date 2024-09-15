
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
// *** clear && ./VanillaNet-cpp -Tr ./Resources/Dataset/mnist_train/ -Te ./Resources/Dataset/mnist_test/ -E 5 -BS 60 -LR 0.5
// *** 
// ***                                             ALL COMANDS IN TWO LINES
// ***
// *** MACOS
// *** rm -r build && mkdir build && cmake -S . -B build
// *** make -C build && clear && ./VanillaNet-cpp -Tr ./Resources/Dataset/mnist_train/ -Te ./Resources/Dataset/mnist_test/ -E 5 -BS 60 -LR 0.5
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

// Complete readme
// Comment aand clear code

int main(int argc, char **argv)
{
    Arguments inputParams;
    int res = parser(inputParams, argc, argv);
    if (res != 0) return res;

    Network net;
    net.addLayer(Layer(784, 128));
    net.addLayer(ActivationLayer(ActivationType::RELU));
    net.addLayer(Layer(128, 10));
    net.addLayer(ActivationLayer(ActivationType::SOFTMAX));

    net.addLossFunction(LossFunction::CROSS_ENTROPY);

    //! Remove after testing
    inputParams.hasWeightsBiases = true;
    // inputParams.WeightsBiasesPath = "./Resources/output/weights/test.json";
    inputParams.WeightsBiasesPath = "./Resources/output/weights/mnist_fc128_relu_fc10_softmax_weights_biases.json";
    // inputParams.WeightsBiasesPath = "./Resources/output/weights/09_13_24/fc128_ReLU_fc10_Sigmoid_09_13_24_20_31_17.json";
    inputParams.learningRate = 0.5;
    inputParams.batchSize = 60;
    inputParams.epochs = 5;
    //!
    
    infoPrinter(inputParams, net);

    std::vector<BiasesWeights> importedWeightsAndBiases;
    weightsBiasExtractor(inputParams, importedWeightsAndBiases);
    net.importWeightsBiases(importedWeightsAndBiases);

    // TRAIN
    networkTrain(net, inputParams);

    // std::vector<std::string> jsonFiles = getJsonFiles("./Resources/output/weights/09_14_24/");
    // ./Resources/output/weights/09_14_24/fc128_ReLU_fc10_Softmax_09_14_24_23_26_55.json
    // weightsNetworkTest(net, inputParams, jsonFiles);

    // TEST
    networkTest(net, inputParams);

    return 0;
}