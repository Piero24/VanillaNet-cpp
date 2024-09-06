#include "network.hpp"


Network::Network() {
    this->loss = 0.0;
    this->lossPrime = 0.0;
}


void Network::addLayer(const Layer& layer)
{
    Layers.push_back(std::make_shared<Layer>(layer));
    standardLayerCount++;
}


void Network::addLayer(const ActivationLayer& activationLayer)
{
    Layers.push_back(std::make_shared<ActivationLayer>(activationLayer));
    activationLayerCount++;
}


void Network::importWeightsBiases(std::vector<BiasesWeights> weightsBiases)
{
    int weightBiasIndex = 0;

    if (weightsBiases.size() != standardLayerCount)
    {
        printf("Error: Number of layers in the network does not match the number of weights and biases provided.\n");
        return;
    }

    for (int i = 0; i < Layers.size(); i++)
    {
        // Identify the type of layer using the polymorphic method getType
        if (Layers[i]->getType() == LayerType::StandardLayer)
        {
            Layers[i]->importWeightsBiases(weightsBiases[weightBiasIndex].weights, weightsBiases[weightBiasIndex].biases);
            weightBiasIndex++;
        }

        if (weightBiasIndex > weightsBiases.size()) break;
    }
}


void Network::setLoss(double loss, double lossPrime)
{
    this->loss = loss;
    this->lossPrime = lossPrime;
}


std::vector<double> Network::forwardPropagation(const std::vector<double>& inputs)
{
    std::vector<double> outputs = inputs;

    for (size_t i = 0; i < Layers.size(); i++)
    {
        outputs = Layers[i]->forwardPass(outputs);
    }

    return outputs;
}