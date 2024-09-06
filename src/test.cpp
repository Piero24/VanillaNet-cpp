#include "test.hpp"


int test(Network &net, Arguments &inputParams)
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
    for (const auto& testResult : testResults)
    {
        correct += testResult.trueValue == testResult.predictedValue;
    }

    std::cout << "Test results: " << std::endl;
    std::cout << "Total correct classifications: " << correct << " out of " << testResults.size() << std::endl;
    std::cout << "Accuracy: " << 100*((double)correct / testResults.size())  << "%" << std::endl;

    // std::cout << "\n\nWrong predictions: " << std::endl;
    // for (const auto& testResult : testResults)
    // {
    //     if (testResult.trueValue != testResult.predictedValue)
    //     {
    //         std::cout << "True value: " << testResult.trueValue << " Predicted value: " << testResult.predictedValue << " Loss: " << testResult.loss << " Image path: " << testResult.imagePath << std::endl;
    //     }
    // }

    return 0;
}