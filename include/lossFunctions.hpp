#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

#include <iostream>
#include <vector>


/**
 * @brief Computes the Mean Squared Error (MSE) loss between two vectors.
 * 
 * The Mean Squared Error (MSE) is a common loss function used in regression tasks.
 * It measures the average of the squares of the differences between the true values 
 * (yTrue) and the predicted values (yPredicted). A lower MSE indicates that the predicted
 * values are closer to the true values.
 * 
 ** Formula: MSE = (1/n) * Σ (yTrue[i] - yPredicted[i])^2
 * 
 * @param yTrue A vector of true/target values. This represents the actual values from the dataset.
 * @param yPredicted A vector of predicted values. These are the values predicted by the model.
 * 
 * @return The Mean Squared Error (MSE) as a double value.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors.
 */
double mse_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


/**
 * @brief Computes the gradient of the Mean Squared Error (MSE) loss function.
 * 
 * The gradient of the Mean Squared Error (MSE) loss function is used in backpropagation to update the 
 * weights and biases of the neural network. It represents the rate of change of the loss with respect 
 * to the predicted values. The gradient is used to adjust the model parameters to minimize the loss.
 * 
 ** Formula: ∂MSE/∂yPredicted[i] = (2/n) * (yPredicted[i] - yTrue[i])
 * 
 * @param yTrue A vector of true/target values. This represents the actual values from the dataset.
 * @param yPredicted A vector of predicted values. These are the values predicted by the model.
 * 
 * @return A vector containing the gradient of the Mean Squared Error (MSE) loss function with respect to 
 *         each predicted value.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors.
 */
std::vector<double> mse_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);



#endif // LOSSFUNCTIONS_HPP
