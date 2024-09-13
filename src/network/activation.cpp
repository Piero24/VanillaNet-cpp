#include "activation.hpp"

std::vector<double> Activation(ActivationType activationFunction, std::vector<double> inputs)
{
    double Z = 0.0;
    double D = 0.0;

    switch (activationFunction)
    {
        case ActivationType::SOFTMAX:
        {
            auto max_it = std::max_element(inputs.begin(), inputs.end());

            // Check if the vector is not empty
            if (max_it != inputs.end())
                D = -*max_it;

            for (int j = 0; j < inputs.size(); j++)
                Z += exp(inputs[j]+D);
            break;
        }
        
        default:
            break;
    }

    for (int i = 0; i < inputs.size(); i++)
    {
        switch (activationFunction)
        {
            case ActivationType::SIGMOID:
                inputs[i] = 1 / (1 + exp(-inputs[i]));
                break;
            
            case ActivationType::SIGMOID_PRIME:
                inputs[i] = inputs[i] * (1 - inputs[i]);
                break;
                
            case ActivationType::RELU:
                inputs[i] = std::max(0.0, inputs[i]);
                break;
            
            case ActivationType::RELU_PRIME:
                if (inputs[i] > 0.00) inputs[i] = 1.0;
                else inputs[i] = 0.0;
                break;
                
            case ActivationType::TANH:
                inputs[i] = tanh(inputs[i]);
                break;
            
            case ActivationType::TANH_PRIME:
                inputs[i] = 1 - pow(tanh(inputs[i]), 2);
                break;
                
            case ActivationType::SOFTMAX:
                // Stable version of the softmax function
                // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
                inputs[i] = exp(inputs[i] + D) / Z;
                break;
            
            // case ActivationType::SOFTMAX_PRIME:
            // https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
            // https://alexcpn.github.io/html/NN/ml/8_backpropogation_full/
                
            default:
                break;
        }
        
    }

    return inputs;
}


std::string ActivationTypeToString(ActivationType activationFunction)
{
    switch (activationFunction)
    {
        case ActivationType::SIGMOID:
            return "Sigmoid";
        
        case ActivationType::SIGMOID_PRIME:
            return "Sigmoid Prime";
        
        case ActivationType::RELU:
            return "ReLU";

        case ActivationType::RELU_PRIME:
            return "ReLU Prime";
        
        case ActivationType::TANH:
            return "Tanh";
        
        case ActivationType::TANH_PRIME:
            return "Tanh Prime";
        
        case ActivationType::SOFTMAX:
            return "Softmax";
        
        default:
            return "None";
    }
}


ActivationType select_dActivation(ActivationType activationFunction)
{
    switch (activationFunction)
    {
        case ActivationType::SIGMOID:
            return ActivationType::SIGMOID_PRIME;
        
        case ActivationType::RELU:
            return ActivationType::RELU_PRIME;
        
        case ActivationType::TANH:
            return ActivationType::TANH_PRIME;
        
        case ActivationType::SOFTMAX:
            return ActivationType::SOFTMAX;
        
        default:
            break;
    }

    printf("[WARNING]: Activation function not supported - Returned the given activation function.\n");
    return activationFunction;

}