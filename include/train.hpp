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
