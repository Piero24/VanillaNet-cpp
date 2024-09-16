#include "test.hpp"


int networkTest(Network &net, Arguments &inputParams)
{
    if (!inputParams.Test)
        return 0;
    
    int correct = 0;
    double averageLoss = 1.0;

    for (int i = 0; i < inputParams.TestDatasetImages.size(); i++)
    {
        VectorLabel vecLabel;
        imageToVectorAndLabel(vecLabel, inputParams.TestDatasetImages[i]);

        std::vector<double> outputOput = net.forwardPropagation(vecLabel.imagePixelVector);
        double lossValue = net.loss(vecLabel.labelVector, outputOput);;
        averageLoss *= lossValue;

        auto max_element_iter = std::max_element(outputOput.begin(), outputOput.end());

        int predictedLabel = 0;
        if (max_element_iter != outputOput.end())
            predictedLabel = std::distance(outputOput.begin(), max_element_iter);
        
        correct += (vecLabel.label == predictedLabel);
        
        printSampleTestResults(inputParams.print, i, correct, inputParams.TestDatasetImages.size(), vecLabel.label, lossValue, predictedLabel);
    }

    averageLoss /= inputParams.TestDatasetImages.size();

    std::string title = " TESTING RESULTS ";
    double acc = 100.0 * ((double)correct / inputParams.TestDatasetImages.size());
    
    if (!inputParams.print)
        finalResultPrinter(acc, averageLoss, correct, inputParams.TestDatasetImages.size(), title);

    if (acc >= inputParams.bestAccuracy)
    {
        inputParams.bestAccuracy = acc;
        inputParams.bestWeightsBiasesPath = inputParams.WeightsBiasesPath;
    }
    else removeJsonFiles({inputParams.WeightsBiasesPath});

    return 0;
}


void weightsNetworkTest(Network &net, Arguments &inputParams, std::vector<std::string> jsonFiles)
{
    for (int i = 0; i < jsonFiles.size(); i++)
    {
        inputParams.WeightsBiasesPath = jsonFiles[i];
        infoPrinter(inputParams, net);

        inputParams.hasWeightsBiases = true;
        std::vector<BiasesWeights> importedWeightsAndBiases;
        weightsBiasExtractor(inputParams, importedWeightsAndBiases);
        net.importWeightsBiases(importedWeightsAndBiases);

        printf(">> Testing model %d/%ld\n", i+1, jsonFiles.size());
        printf("Prev Accuracy: %.2f%%\n", inputParams.bestAccuracy);

        networkTest(net, inputParams);
    }

    std::cout << "Best accuracy: " << inputParams.bestAccuracy << "% with file: " << inputParams.bestWeightsBiasesPath << std::endl;
}


void printSampleTestResults(bool print, int n, int correctImagesCount, int dtSize, int label, double lossValue, int predictedLabel)
{
    if (!print || ((n % 10) != 0)) return;
    std::ostringstream ossAcc;
    ossAcc << std::fixed << std::setprecision(2) << 100.0 * ((double)correctImagesCount / dtSize);

    std::cout << ">>> Sample: " << n << "/" << dtSize;
    std::cout << "     Loss: " << lossValue;
    if (dtSize > 1) std::cout << "     Current Accuracy: " << ossAcc.str();
    std::cout << "%    Predicted: " << predictedLabel;
    std::cout << "     True: " << label << "\n" << std::endl;
}