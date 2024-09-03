#include "lossFunctions.hpp"

double mse_loss(std::vector<double> yTrue, std::vector<double> yPredicted)
{
    double totalSum = 0.0;

    for (int i = 0; i < yTrue.size(); i++)
    {
        totalSum += pow(yTrue[i] - yPredicted[i], 2);
    }

    return totalSum / yTrue.size();
}