#include "network.hpp"


Network::Network()
{
    this->lossValue = 0.0;
    this->lossPrimeValue = std::vector<double>();
    this->output = std::vector<double>();
}


void Network::addLayer(const Layer& layer)
{
    Network::checkSoftmaxLastLayer();
    Layers.push_back(std::make_shared<Layer>(layer));
    standardLayerCount++;
}


void Network::addLayer(const ActivationLayer& activationLayer)
{
    Network::checkSoftmaxLastLayer();
    Layers.push_back(std::make_shared<ActivationLayer>(activationLayer));
    activationLayerCount++;
}


void Network::checkSoftmaxLastLayer()
{
    for (int i = 0; i < Layers.size(); i++)
    {
        if (Layers[i]->getType() == LayerType::ActivationLayer)
        {
            // Use dynamic_cast to access the ActivationLayer
            ActivationLayer* activationLayer = dynamic_cast<ActivationLayer*>(Layers[i].get());
            if (activationLayer != nullptr && activationLayer->activationFunction == ActivationType::SOFTMAX)
            {
                printf("[ERROR]: The softmax activation layer must be the last layer in the network.\n");
                exit(1);
            }
        }
    }
}


void Network::addLossFunction(LossFunction lossFunction)
{
    this->lossFunction = lossFunction;
    this->lossFunctionPrime = select_LossFunction_prime(lossFunction);
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


double Network::loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    if (this->lossFunction == LossFunction::SQUARED_ERROR)
        this->lossValue = squared_error_loss(yTrue, yPredicted);

    else if (this->lossFunction == LossFunction::MEAN_SQUARED_ERROR)
        this->lossValue = mse_loss(yTrue, yPredicted);

    else if (this->lossFunction == LossFunction::CROSS_ENTROPY)
        this->lossValue = binary_cross_entropy_loss(yTrue, yPredicted);

    this->lossPrimeValue = lossPrime(yTrue, yPredicted);
    return this->lossValue;
}


std::vector<double> Network::lossPrime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    if (this->lossFunctionPrime == LossFunctionPrime::SQUARED_ERROR_PRIME)
        this->lossPrimeValue = squared_error_loss_prime(yTrue, yPredicted);

    else if (this->lossFunctionPrime == LossFunctionPrime::MEAN_SQUARED_ERROR_PRIME)
        this->lossPrimeValue = mse_loss_prime(yTrue, yPredicted);

    else if (this->lossFunctionPrime == LossFunctionPrime::CROSS_ENTROPY_PRIME)
        this->lossPrimeValue = binary_cross_entropy_loss_prime(yTrue, yPredicted);

    return this->lossPrimeValue;
}


void Network::setLoss(double loss, const std::vector<double>& lossPrime)
{
    this->lossValue = loss;
    this->lossPrimeValue = lossPrime;
}


std::vector<double> Network::forwardPropagation(const std::vector<double>& inputs)
{
    this->inputs = inputs;
    this->output = inputs;

    for (size_t i = 0; i < Layers.size(); i++)
    {
        this->output = Layers[i]->forwardPass(this->output);
    }

    return this->output;
}


std::vector<BiasesWeights> Network::backwardPropagation(const std::vector<double>& outputError)
{
    int skipSoftmax = 1;
    std::vector<double> error = outputError;
    std::vector<BiasesWeights> weightsBiases;

    ActivationLayer* activationLayer = dynamic_cast<ActivationLayer*>(Layers[Layers.size() - 1].get());
    if (activationLayer != nullptr && activationLayer->activationFunction == ActivationType::SOFTMAX)
        skipSoftmax = 2;

    for (int i = Layers.size() - skipSoftmax; i >= 0; i--)
    {
        BiasesWeights gradients;

        if (Layers[i]->getType() == LayerType::StandardLayer)
        {
            error = Layers[i]->backwardPass(error, gradients.weights, gradients.biases);

            gradients.LayerIndex = standardLayerCount - weightsBiases.size();
            gradients.BiasName = "fc" + std::to_string(gradients.LayerIndex) + ".bias";
            gradients.WeightsName = "fc" + std::to_string(gradients.LayerIndex) + ".weight";

            weightsBiases.push_back(gradients);
        }
        else error = Layers[i]->backwardPass(error, gradients.weights, gradients.biases);
    }

    return weightsBiases;
}


void Network::updateWeightsBiases(const std::vector<std::vector<BiasesWeights>>& accumulatedGrad, double learningRate)
{
    std::vector<BiasesWeights> average = calculateAverageGradients(accumulatedGrad);
    std::reverse(average.begin(), average.end());

    // for (int i = 0; i < average.size(); i++)
    // {
    //     for (int j = 0; j < average[i].biases.size(); j++)
    //     {
    //         printf("average[%d].biases[%d]: %.6f\n", i, j, average[i].biases[j]);
    //     }

    //     for (int j = 0; j < average[i].weights.size(); j++)
    //     {
    //         for (int k = 0; k < average[i].weights[j].size(); k++)
    //         {
    //             printf("average[%d].weights[%d][%d]: %.6f\n", i, j, k, average[i].weights[j][k]);
    //         }
    //     }
    // }

    int averageIndex = 0;
    if (average.size() == 0) return;

    if (average.size() != standardLayerCount)
    {
        printf("Error: Number of layers in the network does not match the number of weights and biases provided.\n");
        return;
    }

    for (int i = 0; i < Layers.size(); i++)
    {
        // Identify the type of layer using the polymorphic method getType
        if (Layers[i]->getType() == LayerType::StandardLayer)
        {
            Layers[i]->Layer::updateWeightsBiases(learningRate, average[averageIndex].weights, average[averageIndex].biases);
            averageIndex++;
        }

        if (averageIndex > average.size()) break;
    }
}


std::vector<BiasesWeights> Network::calculateAverageGradients(const std::vector<std::vector<BiasesWeights>>& accumulatedGrad)
{
    std::vector<BiasesWeights> average = accumulatedGrad[0];
    int batchSize = accumulatedGrad.size();

    for (int i = 1; i < accumulatedGrad.size(); i++)
    {
        for (int j = 0; j < accumulatedGrad[i].size(); j++)
        {
            for (int k = 0; k < accumulatedGrad[i][j].biases.size(); k++)
            {
                average[j].biases[k] += accumulatedGrad[i][j].biases[k];
            }

            for (int k = 0; k < accumulatedGrad[i][j].weights.size(); k++)
            {
                for (int l = 0; l < accumulatedGrad[i][j].weights[k].size(); l++)
                {
                    average[j].weights[k][l] += accumulatedGrad[i][j].weights[k][l];
                }
            }
        }
    }

    for (int i = 0; i < average.size(); i++)
    {
        for (int j = 0; j < average[i].biases.size(); j++)
        {
            average[i].biases[j] /= batchSize;
        }

        for (int j = 0; j < average[i].weights.size(); j++)
        {
            for (int k = 0; k < average[i].weights[j].size(); k++)
            {
                average[i].weights[j][k] /= batchSize;
            }
        }
    }

    return average;
}