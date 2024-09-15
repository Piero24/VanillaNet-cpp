#include "lossFunctions.hpp"


double mse_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    assert(yTrue.size() == yPredicted.size());
    double totalSum = 0.0;

    for (int i = 0; i < yTrue.size(); i++)
        totalSum += pow(yTrue[i] - yPredicted[i], 2);

    return totalSum / yTrue.size();
}


std::vector<double> mse_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    assert(yTrue.size() == yPredicted.size());
    std::vector<double> gradient;
    int n = yTrue.size();
    
    for (int i = 0; i < n; i++)
    {
        // Derivative of MSE with respect to each predicted value
        double grad = (2.0 / n) * (yPredicted[i] - yTrue[i]);
        gradient.push_back(grad);
    }
    
    return gradient;
}


double squared_error_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    assert(yTrue.size() == yPredicted.size());
    double loss = 0.0;
    
    for (size_t i = 0; i < yTrue.size(); ++i)
        // Squared Error: L = (yTrue - yPredicted)^2
        loss += pow(yTrue[i] - yPredicted[i], 2);
    
    return loss;
}


std::vector<double> squared_error_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    assert(yTrue.size() == yPredicted.size());
    std::vector<double> derivative(yTrue.size());
    
    for (size_t i = 0; i < yTrue.size(); ++i)
        // Derivative: dL/dy_pred = 2 * (yPredicted - yTrue)
        derivative[i] = 2 * (yPredicted[i] - yTrue[i]);
    
    return derivative;
}


double binary_cross_entropy_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    assert(yTrue.size() == yPredicted.size());
    double loss = 0.0;
    const double epsilon = 1e-12; // Small value to prevent log(0)
    
    for (size_t i = 0; i < yTrue.size(); ++i)
    {
        // Clamping the predictions to prevent log(0)
        double yPred = std::min(std::max(yPredicted[i], epsilon), 1.0 - epsilon);
        loss += yTrue[i] * log(yPred) + (1 - yTrue[i]) * log(1 - yPred);
    }
    
    return -loss;
}


std::vector<double> binary_cross_entropy_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    assert(yTrue.size() == yPredicted.size());
    std::vector<double> derivative(yTrue.size());
    
    for (size_t i = 0; i < yTrue.size(); ++i)
        // Derivative: dL/dy_pred = y_pred - y_true
        derivative[i] = yPredicted[i] - yTrue[i];
    
    return derivative;
}


LossFunctionPrime select_LossFunction_prime(LossFunction lossFunction)
{
    switch (lossFunction)
    {
        case LossFunction::MEAN_SQUARED_ERROR:
            return LossFunctionPrime::MEAN_SQUARED_ERROR_PRIME;

        case LossFunction::SQUARED_ERROR:
            return LossFunctionPrime::SQUARED_ERROR_PRIME;
        
        case LossFunction::CROSS_ENTROPY:
            return LossFunctionPrime::CROSS_ENTROPY_PRIME;
        
        default:
            printf("[WARNING]: Loss function not supported. Returning INVALID loss function prime.\n");
            return LossFunctionPrime::INVALID;
    }
}


std::vector<std::string> lossFunctionTypeToString(LossFunction lossFunction)
{
    switch (lossFunction)
    {
        case LossFunction::MEAN_SQUARED_ERROR:
            return {"Mean Squared Error", "Mean Squared Error Prime"};
        
        case LossFunction::SQUARED_ERROR:
            return {"Squared Error", "Squared Error Prime"};
        
        case LossFunction::CROSS_ENTROPY:
            return {"Cross Entropy", "Cross Entropy Prime"};
        
        default:
            return {"None", "None"};
    }
}