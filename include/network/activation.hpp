#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

std::vector<double> Activation(ActivationType activationFunction, std::vector<double> inputs);

std::string ActivationTypeToString(ActivationType activationFunction);

#endif // ACTIVATION_HPP
