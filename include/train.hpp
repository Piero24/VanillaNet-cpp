#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <algorithm>
#include <random>

#include "toolkit.hpp"
#include "network.hpp"
#include "lossFunctions.hpp"
#include "saveToJson.hpp"
#include "printer.hpp"


/**
 * @brief Holds the result of a single training sample,
 *        including the true and predicted values, loss, and metadata.
 */
struct TrainResult
{
    int trueValue;           ///< The true label or value of the input sample (e.g., class label for classification).
    int predictedValue;      ///< The predicted label or value from the network after forward propagation.
    double loss;             ///< The calculated loss for this sample (e.g., squared error loss).
    std::string imagePath;   ///< The file path of the input image used for this training sample.
    int epoch;               ///< The epoch during which this sample was processed.
    int batch;               ///< The batch within the epoch that contained this sample.
};


/**
 * @brief Trains the neural network using the training dataset, performing forward and backward passes, 
 *        calculating loss, and updating weights and biases.
 * 
 * @param net The neural network to be trained.
 * @param inputParams The training parameters, including the dataset, epochs, batch size, and learning rate.
 * @return int Returns 0 upon successful training completion.
 */
int networkTrain(Network &net, Arguments &inputParams);


/**
 * @brief Splits the input vector into batches of a specified size.
 * 
 * @param inputVec The vector containing the input data (e.g., image paths).
 * @param batchSize The size of each batch.
 * @return std::vector<std::vector<std::string>> A 2D vector where each inner vector is a batch of input data.
 */
std::vector<std::vector<std::string>> splitIntoBatches(const std::vector<std::string>& inputVec, int batchSize);


#endif // TRAIN_HPP
