#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <algorithm>
#include <random>

#include "toolkit.hpp"
#include "network.hpp"
#include "lossFunctions.hpp"
#include "saveToJson.hpp"


struct TrainResult
{
    int trueValue;
    int predictedValue;
    double loss;
    std::string imagePath;
    int epoch;
    int batch;
};


int networkTrain(Network &net, Arguments &inputParams, int epochs, int batchSize, double learningRate);

std::vector<std::vector<std::string>> splitIntoBatches(const std::vector<std::string>& inputVec, int batchSize);

#endif // TRAIN_HPP
