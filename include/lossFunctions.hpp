#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>


// https://mccormickml.com/2014/03/04/gradient-descent-derivation/


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


/**
 * @brief Computes the Squared Error Loss between two vectors.
 * 
 * The Squared Error Loss measures the average of the squares of the differences between 
 * the true values (yTrue) and the predicted values (yPredicted). It is similar to 
 * Mean Squared Error but does not divide by the number of samples. A lower loss indicates 
 * that the predictions are closer to the true values.
 * 
 ** Formula: Squared Error Loss = Σ (yTrue[i] - yPredicted[i])^2
 * 
 * @param yTrue A vector of true/target values. This represents the actual values from the dataset.
 * @param yPredicted A vector of predicted values. These are the values predicted by the model.
 * 
 * @return The Squared Error Loss as a double value.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors.
 */
double squared_error_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


/**
 * @brief Computes the gradient of the Squared Error Loss function.
 * 
 * The gradient of the Squared Error Loss function is used in backpropagation to update the 
 * weights and biases of the neural network. It indicates how the loss changes with respect 
 * to the predicted values. This information is essential for adjusting model parameters to minimize the loss.
 * 
 ** Formula: ∂SquaredError/∂yPredicted[i] = 2 * (yPredicted[i] - yTrue[i])
 * 
 * @param yTrue A vector of true/target values. This represents the actual values from the dataset.
 * @param yPredicted A vector of predicted values. These are the values predicted by the model.
 * 
 * @return A vector containing the gradient of the Squared Error Loss function with respect to 
 *         each predicted value.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors.
 */
std::vector<double> squared_error_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


/**
 * @brief Computes the Binary Cross-Entropy Loss between two vectors.
 * 
 * The Binary Cross-Entropy Loss measures the performance of a classification model whose output 
 * is a probability value between 0 and 1. It quantifies the difference between two probability distributions 
 * – the true distribution (yTrue) and the predicted distribution (yPredicted). A lower loss indicates better 
 * performance.
 * 
 ** Formula: Binary Cross-Entropy Loss = - (1/n) * Σ [yTrue[i] * log(yPredicted[i]) + (1 - yTrue[i]) * log(1 - yPredicted[i])]
 * 
 * @param yTrue A vector of true binary target values (0 or 1). These represent the actual values from the dataset.
 * @param yPredicted A vector of predicted probabilities (between 0 and 1). These are the values predicted by the model.
 * 
 * @return The Binary Cross-Entropy Loss as a double value.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors. Additionally, yPredicted values should be between 0 and 1.
 */
double binary_cross_entropy_loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


/**
 * @brief Computes the gradient of the Binary Cross-Entropy Loss function.
 * 
 * The gradient of the Binary Cross-Entropy Loss function is used in backpropagation to adjust 
 * the weights and biases of the neural network. It indicates how the loss changes with respect 
 * to the predicted probabilities. This information is crucial for minimizing the loss during training.
 * 
 ** Formula: ∂BinaryCrossEntropy/∂yPredicted[i] = (yPredicted[i] - yTrue[i]) / [yPredicted[i] * (1 - yPredicted[i])]
 * 
 * @param yTrue A vector of true binary target values (0 or 1). These represent the actual values from the dataset.
 * @param yPredicted A vector of predicted probabilities (between 0 and 1). These are the values predicted by the model.
 * 
 * @return A vector containing the gradient of the Binary Cross-Entropy Loss function with respect to 
 *         each predicted probability.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors. Additionally, yPredicted values should be between 0 and 1 to avoid 
 *       division by zero.
 */
std::vector<double> binary_cross_entropy_loss_prime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


#endif // LOSSFUNCTIONS_HPP
