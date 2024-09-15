#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>


// https://mccormickml.com/2014/03/04/gradient-descent-derivation/


/**
 * @brief Enum class representing different types of loss functions used in neural networks.
 * 
 * This enum class defines various loss functions that can be utilized during the 
 * training and evaluation of a neural network. Each loss function is associated 
 * with a specific method for measuring the discrepancy between the predicted output 
 * and the true labels.
 * 
 * - **MEAN_SQUARED_ERROR**: The average of the squared differences between 
 *   predicted values and true values. Commonly used in regression tasks.
 * 
 * - **SQUARED_ERROR**: The sum of squared differences, which is similar to 
 *   mean squared error but not averaged. 
 * 
 * - **CROSS_ENTROPY**: A measure of the difference between two probability distributions, 
 *   commonly used in classification tasks.
 * 
 * - **INVALID**: Indicates an unsupported or unrecognized loss function. This can be 
 *   used for error handling and validation.
 */
enum class LossFunction {
    MEAN_SQUARED_ERROR, ///< Average squared error loss function.
    SQUARED_ERROR,      ///< Sum of squared error loss function.
    CROSS_ENTROPY,      ///< Cross-entropy loss function for classification.
    INVALID             ///< Indicates an unsupported loss function.
};


/**
 * @brief Enum class representing the derivatives (primes) of different loss functions.
 * 
 * This enum class defines the derivatives corresponding to each loss function 
 * defined in the `LossFunction` enum. The derivatives are essential for 
 * calculating gradients during the backpropagation phase of neural network training.
 * 
 * - **MEAN_SQUARED_ERROR_PRIME**: Derivative of the mean squared error loss function.
 * 
 * - **SQUARED_ERROR_PRIME**: Derivative of the sum of squared error loss function.
 * 
 * - **CROSS_ENTROPY_PRIME**: Derivative of the cross-entropy loss function.
 * 
 * - **INVALID**: Indicates an unsupported or unrecognized loss function derivative. 
 *   This can be used for error handling and validation.
 */
enum class LossFunctionPrime {
    MEAN_SQUARED_ERROR_PRIME,  ///< Derivative of the mean squared error loss function.
    SQUARED_ERROR_PRIME,       ///< Derivative of the sum of squared error loss function.
    CROSS_ENTROPY_PRIME,       ///< Derivative of the cross-entropy loss function.
    INVALID                    ///< Indicates an unsupported loss function prime.
};


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


/**
 * @brief Maps a loss function to its corresponding derivative (prime) function.
 * 
 * This function takes a specified loss function and returns the corresponding 
 * derivative (gradient) function, which will be used during backpropagation.
 * 
 * - For each loss function (e.g., `MEAN_SQUARED_ERROR`, `SQUARED_ERROR`, `CROSS_ENTROPY`), 
 *   there is an associated derivative (e.g., `MEAN_SQUARED_ERROR_PRIME`).
 * 
 * @param lossFunction The loss function for which the derivative is required.
 * @return The corresponding loss function derivative (as `LossFunctionPrime`), or 
 *         `INVALID` if the loss function is not supported.
 * 
 * @warning If the loss function is not recognized, a warning message is printed, 
 * and the function returns `LossFunctionPrime::INVALID`.
 */
LossFunctionPrime select_LossFunction_prime(LossFunction lossFunction);


/**
 * @brief Converts a loss function and its derivative to human-readable string representations.
 * 
 * This function converts a given loss function and its corresponding derivative 
 * (prime) function into a pair of human-readable strings. These strings can be 
 * used for logging, debugging, or user display purposes.
 * 
 * - For supported loss functions (e.g., `MEAN_SQUARED_ERROR`, `SQUARED_ERROR`, 
 *   `CROSS_ENTROPY`), it returns a vector of two strings:
 *     - The first string represents the name of the loss function.
 *     - The second string represents the name of the loss function's derivative (prime).
 * 
 * @param lossFunction The loss function for which a string representation is required.
 * @return A vector of strings representing the loss function and its derivative.
 * 
 * @note If the loss function is not recognized, the function returns "None" for both the loss function and its derivative.
 */
std::vector<std::string> lossFunctionTypeToString(LossFunction lossFunction);


#endif // LOSSFUNCTIONS_HPP
