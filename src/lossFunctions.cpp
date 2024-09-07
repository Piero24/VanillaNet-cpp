#include "lossFunctions.hpp"


double mse_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
    double totalSum = 0.0;

    for (int i = 0; i < yTrue.size(); i++)
    {
        totalSum += pow(yTrue[i] - yPredicted[i], 2);
    }

    return totalSum / yTrue.size();
}


std::vector<double> mse_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted)
{
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
