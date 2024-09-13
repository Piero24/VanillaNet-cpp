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
    
    for (size_t i = 0; i < yTrue.size(); ++i)
        // Binary cross-entropy: L = -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)
        loss += yTrue[i] * log(yPredicted[i]) + (1 - yTrue[i]) * log(1 - yPredicted[i]);
    
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
