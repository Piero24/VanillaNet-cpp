
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


int main(int argc, char **argv)
{
    Arguments inputParams;
    int res = parser(inputParams, argc, argv);
    if (res != 0)
    {
        return res;
    }

    infoPrinter(inputParams);

    std::vector<BiasesWeights> importedWeightsAndBiases;

    Network net;
    net.addLayer(Layer(784, 128));
    net.addLayer(ActivationLayer(ActivationType::RELU));
    net.addLayer(Layer(128, 10));
    net.addLayer(ActivationLayer(ActivationType::SOFTMAX));

    //! Remove after testing
    inputParams.hasWeightsBiases = true;

    if (inputParams.hasWeightsBiases)
    {
        // importedWeightsAndBiases = parseJSON("./Resources/output/weights/test.json");
        importedWeightsAndBiases = parseJSON("./Resources/output/weights/mnist_fc128_relu_fc10_log_softmax_weights_biases.json");
        // importedWeightsAndBiases = parseJSON("./Resources/output/weights/mnist_fc128_relu_fc10_sigmoid_weights_biases.json");
        // jsonValuePrinter(importedWeightsAndBiases);
        net.importWeightsBiases(importedWeightsAndBiases);
    }

    // TRAIN
    train(inputParams);

    // TEST
    test(net, inputParams);
    
    return 0;
}