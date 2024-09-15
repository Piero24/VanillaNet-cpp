#ifndef TEST_HPP
#define TEST_HPP

#include "network.hpp"
#include "toolkit.hpp"
#include "lossFunctions.hpp"
#include "printer.hpp"
#include "tester.hpp"

/**
 * @brief Holds the result of a single test sample, 
 *        including the true and predicted values, loss, and image path.
 */
struct TestResult
{
    int trueValue;           ///< The true label or value of the test sample (e.g., class label for classification).
    int predictedValue;      ///< The predicted label or value from the network after forward propagation.
    double loss;             ///< The calculated loss for this sample (e.g., mean squared error loss).
    std::string imagePath;   ///< The file path of the input image used for this test sample.
};


/**
 * @brief Tests the neural network on a given test dataset, performing forward propagation, calculating loss, 
 *        and determining the accuracy and average loss across the test set.
 * 
 * @param net The neural network to be tested.
 * @param inputParams The testing parameters, including the test dataset.
 * @return int Returns 0 upon successful testing completion.
 */
int networkTest(Network &net, Arguments &inputParams);


void weightsNetworkTest(Network &net, Arguments &inputParams, std::vector<std::string> jsonFiles);


#endif // TEST_HPP
