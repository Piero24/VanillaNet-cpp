#include "layer.hpp"


Layer::Layer(int numberOfNeurons, std::vector<double> inputs)
{
    this->inputs = inputs;
    this->numberOfNeurons = numberOfNeurons;
    this->neurons = Layer::initializeNeurons(numberOfNeurons, inputs);
    this->outputs = Layer::layerOutput(neurons);

}


std::vector<Neuron> Layer::initializeNeurons(int numberOfNeurons, std::vector<double> inputs)
{
    std::vector<Neuron> neurons;
    neurons.reserve(numberOfNeurons);

    for (int i = 0; i < numberOfNeurons; i++) {
        neurons.emplace_back(inputs);
    }

    return neurons;
}


std::vector<double> Layer::layerOutput(std::vector<Neuron> neurons)
{
    std::vector<double> out(neurons.size());

    for (int i = 0; i < neurons.size(); i++)
    {
        out[i] = neurons[i].output;
    }   

    return out;
}