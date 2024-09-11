#include "layer.hpp"


Layer::Layer(int inputSize, int outputSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->neurons = Layer::initializeNeurons(inputSize, outputSize);
}


int Layer::getLayerSize()
{
    return outputSize;
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

        neurons[i].setWeights(weights[i]);
        neurons[i].setBias(biases[i]);
    }

}


void Layer::saveWeightsBiases(std::vector<std::vector<double>>& weights, std::vector<double>& biases)
{
    for (int i = 0; i < neurons.size(); i++)
    {
        weights.push_back(neurons[i].weights);
        biases.push_back(neurons[i].bias);
    }
}


std::vector<double> Layer::forwardPass(std::vector<double> inputs)
{
    std::vector<double> outputs;
    outputs.reserve(neurons.size());

    for (int i = 0; i < neurons.size(); i++)
    {
        outputs.push_back(neurons[i].getOutput(inputs));
    }

    this->outputs = outputs;
    return outputs;
}


std::vector<double> Layer::backwardPass(std::vector<double>& error, std::vector<std::vector<double>>& weights, std::vector<double>& biases)
{
    for (int i = 0; i < neurons.size(); i++)
    {
        
    }

}


std::vector<Neuron> Layer::initializeNeurons(int inputSize, int outputSize)
{
    std::vector<Neuron> neurons;
    neurons.reserve(outputSize);

    for (int i = 0; i < outputSize; i++)
    {
        neurons.emplace_back(inputSize);
    }

    return neurons;
}


ActivationLayer::ActivationLayer(ActivationType activationFunction) : Layer(0, 0)
{
    this->activationFunction = activationFunction;
}


std::vector<double> ActivationLayer::forwardPass(std::vector<double> inputs)
{
    this->outputs = Activation(this->activationFunction, inputs);
    return this->outputs;
}


std::vector<double> ActivationLayer::backwardPass(std::vector<double>& error, std::vector<std::vector<double>>& weights, std::vector<double>& biases)
{
    ActivationType sel_dAct = select_dActivation(this->activationFunction);
    return Activation(sel_dAct, error);
}