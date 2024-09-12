#include "network.hpp"


Network::Network() {
    this->loss = 0.0;
    this->lossPrime = 0.0;
    this->output = std::vector<double>();
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

    if (weightsBiases.size() == 0) return;

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


std::vector<BiasesWeights> Network::saveWeightsBiases()
{
    std::vector<BiasesWeights> weightsBiases;
    int idx = 1;

    for (int i = 0; i < Layers.size(); i++)
    {
        // Identify the type of layer using the polymorphic method getType
        if (Layers[i]->getType() == LayerType::StandardLayer)
        {
            BiasesWeights bw;
            Layers[i]->saveWeightsBiases(bw.weights, bw.biases);

            bw.LayerIndex = idx++;
            bw.BiasName = "fc" + std::to_string(bw.LayerIndex) + ".bias";
            bw.WeightsName = "fc" + std::to_string(bw.LayerIndex) + ".weight";

            weightsBiases.push_back(bw);
        }
    }

    return weightsBiases;
}


void Network::setLoss(double loss, double lossPrime)
{
    this->loss = loss;
    this->lossPrime = lossPrime;
}


std::vector<double> Network::forwardPropagation(const std::vector<double>& inputs)
{
    this->output = inputs;

    for (size_t i = 0; i < Layers.size(); i++)
    {
        this->output = Layers[i]->forwardPass(this->output);
    }

    return this->output;
}


std::vector<BiasesWeights> Network::backwardPropagation(const std::vector<double>& outputError)
{
    std::vector<double> error = outputError;
    std::vector<BiasesWeights> weightsBiases;

    for (int i = Layers.size() - 1; i >= 0; i--)
    {
        BiasesWeights gradients;

        if (Layers[i]->getType() == LayerType::StandardLayer)
        {
            Layers[i]->backwardPass(error, gradients.weights, gradients.biases);

            gradients.LayerIndex = i + 1;
            gradients.BiasName = "fc" + std::to_string(gradients.LayerIndex) + ".bias";
            gradients.WeightsName = "fc" + std::to_string(gradients.LayerIndex) + ".weight";

            weightsBiases.push_back(gradients);
        }
        else error = Layers[i]->backwardPass(error, gradients.weights, gradients.biases);
    }
}