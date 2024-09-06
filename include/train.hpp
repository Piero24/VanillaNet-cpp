#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <algorithm>
#include <random>

#include "toolkit.hpp"


int networkTrain(Arguments &inputParams, int epochs, int batchSize, double learningRate);

std::vector<std::vector<std::string>> splitIntoBatches(const std::vector<std::string>& inputVec, int batchSize);

#endif // TRAIN_HPP
