#include "activation.hpp"


std::vector<double> Activation(ActivationType activationFunction, std::vector<double> inputs)
{

    double Z = 0.0;

    switch (activationFunction)
    {
        case ActivationType::SOFTMAX: {
            for (int j = 0; j < inputs.size(); j++)
            {
                Z += exp(inputs[j]);
            }
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
                break;  // Add break here
                
            case ActivationType::RELU:
                inputs[i] = std::max(0.0, inputs[i]);
                break;  // Add break here
                
            case ActivationType::TANH:
                inputs[i] = tanh(inputs[i]);
                break;  // Add break here
                
            case ActivationType::SOFTMAX:
                inputs[i] = exp(inputs[i]) / Z;
                break;  // Add break here
                
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
        
        case ActivationType::RELU:
            return "ReLU";
        
        case ActivationType::TANH:
            return "Tanh";
        
        case ActivationType::SOFTMAX:
            return "Softmax";
        
        default:
            return "None";
    }
}