#include "neuron2.hpp"


std::default_random_engine Neuron::re(static_cast<unsigned long>(std::time(nullptr)));


Neuron::Neuron(std::vector<double> inputs)
{
    this->inputs = inputs;  // Store the input values
    this->numberOfInputs = inputs.size();  // Store the number of inputs
    this->bias = Neuron::initializeBias();  // Initialize bias
    this->weights = Neuron::initializeWeights(numberOfInputs);  // Initialize weights
    this->output = Neuron::getOutput(inputs, this->weights, this->bias);  // Calculate initial output
}


std::vector<double> Neuron::initializeWeights(int numberOfInputs)
{
    std::vector<double> weights(numberOfInputs);
    std::uniform_real_distribution<double> unif(-0.5, 0.5);

    for (int i = 0; i < numberOfInputs; i++)  // Use numberOfInputs directly
    {
        weights[i] = unif(re);
    }

    return weights;
}


double Neuron::initializeBias()
{
    std::uniform_real_distribution<double> unif(-0.5, 0.5);
    return unif(re);
}


void Neuron::setWeights(double weight, int position)
{
    weights[position] = weight;
}


void Neuron::setBias(double weight)
{
    bias = weight;
}


double Neuron::getOutput(std::vector<double> inputs, std::vector<double> weights, int bias)
{
    double result = 0.0;

    for (int i = 0; i < inputs.size(); i++)
    {
        result += inputs[i] * weights[i];
    }

    return result += bias;
}


void Neuron::recalculateOutput() {
    this->output = Neuron::getOutput(this->inputs, this->weights, this->bias);
}