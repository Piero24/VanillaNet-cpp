#include "layer2.hpp"


// Constructor implementation
Layer::Layer(int numberOfNeurons, std::vector<double> inputs)
{
    this->inputs = inputs;  // Store the input values for the layer
    this->numberOfNeurons = numberOfNeurons;  // Store the number of neurons
    this->neurons = Layer::initializeNeurons(numberOfNeurons, inputs);  // Initialize neurons
    this->outputs = Layer::layerOutput(neurons);  // Calculate the layer's output based on the neurons
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


void Layer::importWeightsBiases(std::vector<std::vector<double>> weights, std::vector<double> biases)
{
    // printf("Weights size: %ld, Biases size: %ld, Neurons size: %ld\n", weights.size(), biases.size(), neurons.size());
    
    if (weights.size() != neurons.size() || biases.size() != neurons.size())
    {
        printf("Error: Weights vector size: %ld or Biases size: %ld does not match the number of neurons: %ld\n", weights.size(), biases.size(), neurons.size());
        return;
    }
    
    for (int i = 0; i < neurons.size(); i++)
    {
        //printf("%d) Weights size: %ld, Neuron size: %ld\n", i+1, weights[i].size(), neurons[i].weights.size());
        
        if (weights[i].size() != neurons[i].weights.size())
        {
            printf("Error: Weights size: %ld does not match the number of inputs: %ld\n", weights[i].size(), neurons[i].weights.size());
            return;
        }

        for (int j = 0; j < weights[i].size(); j++)
        {
            neurons[i].setWeights(weights[i][j], j);
        }

        neurons[i].setBias(biases[i]);
        neurons[i].recalculateOutput();
    }

    this->outputs = Layer::layerOutput(neurons);
}