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


/**
 * @brief Tests multiple sets of weights and biases on the neural network and evaluates the performance.
 * 
 * This function is used to load and test multiple sets of weights and biases from different JSON files on 
 * the provided neural network (`net`). The goal is to evaluate the performance of the network using 
 * each set of weights and biases and track the best-performing model.
 * 
 * Steps involved:
 * - Iterate over the list of JSON files (`jsonFiles`), each containing a set of weights and biases.
 * - For each file, update the `WeightsBiasesPath` in `inputParams` and print information about the network.
 * - Set the flag `inputParams.hasWeightsBiases` to indicate that weights and biases are being imported.
 * - Extract the weights and biases using the `weightsBiasExtractor` function and import them into the network 
 *   using `net.importWeightsBiases()`.
 * - Print the current model number, previous accuracy, and test the network using `networkTest()`.
 * 
 * After testing all the models, the function prints the best accuracy and the corresponding file that produced it.
 * 
 * @param net Reference to the neural network object.
 * @param inputParams Reference to the `Arguments` structure containing input parameters and network settings.
 * @param jsonFiles A vector of strings, each containing the path to a JSON file with weights and biases to be tested.
 * 
 * @note The function tracks the best accuracy (`inputParams.bestAccuracy`) and the corresponding weights and biases file (`inputParams.bestWeightsBiasesPath`) throughout the testing process.
 */
void weightsNetworkTest(Network &net, Arguments &inputParams, std::vector<std::string> jsonFiles);


#endif // TEST_HPP
