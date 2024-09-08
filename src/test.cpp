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

    printf("\n");
    printCentered(" TESTING RESULTS ", '*');
    printf("\nCorrectly Classified:  %d/%ld\t\t\t\t\t\t\t\t\t\t", correct, testResults.size());
    printf("Accuracy:  %.2f%%\t\t\t\t\t\t\t\t\t\t", 100.0 * ((double)correct / testResults.size()));
    printf("Average loss:  %.5f\t\t", averageLoss);

    printf("\n\n");
    printHorizontalLine('*');

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