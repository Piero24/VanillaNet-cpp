#include "neuron.hpp"


std::default_random_engine Neuron::re(static_cast<unsigned long>(std::time(nullptr)));


Neuron::Neuron(int inputSize)
{
    this->bias = Neuron::initializeBias();
    this->inputSize = inputSize;
    this->weights = Neuron::initializeWeights(inputSize);
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

    for (int i = 0; i < inputs.size(); i++)
    {
        result += inputs[i] * weights[i];
    }

    return result += bias;
}


double Neuron::initializeBias()
{
    std::uniform_real_distribution<double> unif(-0.5, 0.5);
    return unif(re);
}


std::vector<double> Neuron::initializeWeights(int inputSize)
{
    std::vector<double> weightsVec(inputSize);
    std::uniform_real_distribution<double> unif(-0.5, 0.5);

    for (int i = 0; i < inputSize; i++)
    {
        weightsVec[i] = unif(re);
    }

    return weightsVec;
}