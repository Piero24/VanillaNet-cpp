#include "test.hpp"


int networkTest(Network &net, Arguments &inputParams)
{
    std::vector<TestResult> testResults;

    if (!inputParams.Test)
        return 0;

    for (const auto& imagePath : inputParams.TestDatasetImages)
    {
        TestResult test;

        VectorLabel vecLabel;
        imageToVectorAndLabel(vecLabel, imagePath);
        test.trueValue = vecLabel.label;
        test.imagePath = imagePath;

        std::vector<double> outputOput = net.forwardPropagation(vecLabel.imagePixelVector);
        test.loss = mse_loss(vecLabel.labelVector, outputOput);

        auto max_element_iter = std::max_element(outputOput.begin(), outputOput.end());

        if (max_element_iter != outputOput.end())
            test.predictedValue = std::distance(outputOput.begin(), max_element_iter);
        
        testResults.push_back(test);
    }

    int correct = 0;
    double averageLoss = 0.0;
    for (const auto& testResult : testResults)
    {
        correct += testResult.trueValue == testResult.predictedValue;
        averageLoss += testResult.loss;
    }
    averageLoss /= testResults.size();

    std::string title = " TESTING RESULTS ";
    double acc = 100.0 * ((double)correct / testResults.size());
    
    finalResultPrinter(acc, averageLoss, correct, testResults.size(), title);


    // std::cout << "\n\nWrong predictions: " << std::endl;
    // for (const auto& testResult : testResults)
    // {
    //     if (testResult.trueValue != testResult.predictedValue)
    //     {
    //         std::cout << "True value: " << testResult.trueValue << " Predicted value: " << testResult.predictedValue << " Loss: " << testResult.loss << " Image path: " << testResult.imagePath << std::endl;
    //     }
    // }

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