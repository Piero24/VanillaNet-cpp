
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
        // jsonValuePrinter(importedWeightsAndBiases);
        net.importWeightsBiases(importedWeightsAndBiases);
    }

    

    // --------------------------------------------------------------------------------------------






    if (inputParams.Train)
    {
        // TODO: Training process        
    }







    // TEST
    if (inputParams.Test)
    {
        for (const auto& imagePath : inputParams.TestDatasetImages)
        {
            VectorLabel vecLabel;
            imageToVectorAndLabel(vecLabel, imagePath);

            std::vector<double> outputOput = net.forwardPropagation(vecLabel.imagePixelVector);

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
            double loss = mse_loss(vecLabel.labelVector, outputOput);

            printf("True value: %d, Predicted Value: %d with probability: %f, Loss: %f\n", vecLabel.label, maxIndex, maxVal, loss);

        }
    }

    return 0;
}