#include "neuron.hpp"

std::default_random_engine Neuron::re(static_cast<unsigned long>(time(nullptr)));


Neuron::Neuron(int inputSize, int outputSize)
{
    this->bias = Neuron::initializeBias();
    this->inputSize = inputSize;
    this->weights = Neuron::initializeWeights(inputSize, outputSize);
}


void Neuron::setWeights(std::vector<double> weight)
{
    this->weights = weight;
    this->inputSize = weight.size();
}


void Neuron::setBias(double bias)
{
    this->bias = bias;
}


double Neuron::getOutput(std::vector<double> inputs)
{
    double result = 0.0;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        result += inputs[i] * weights[i];
    }

    return result += bias;
}


double Neuron::initializeBias()
{
    return 0.0;
}


std::vector<double> Neuron::standardInitializeWeights(int inputSize)
{
    std::vector<double> weightsVec(inputSize);
    std::uniform_real_distribution<double> unif(-0.5, 0.5);

    for (int i = 0; i < inputSize; i++)
    {
        weightsVec[i] = unif(re);
    }

    return weightsVec;
}


std::vector<double> Neuron::initializeWeights(int inputSizeLayer, int outputSizeLayer)
{
    std::vector<double> weightsVec(inputSizeLayer);

    // Calculate the limit for Glorot initialization
    double limit = std::sqrt(6.0 / (inputSizeLayer + outputSizeLayer));
    std::uniform_real_distribution<double> unif(-limit, limit);

    for (int i = 0; i < inputSizeLayer; i++) {
        weightsVec[i] = unif(re);
    }

    return weightsVec;
}