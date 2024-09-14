#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>


/**
 * @brief Enumeration of available activation functions for neural networks.
 * 
 * This enum class defines different types of activation functions that can be applied
 * to neurons in a neural network. Activation functions introduce non-linearity, allowing
 * neural networks to approximate complex functions. Each type has a different mathematical
 * function and use case:
 * 
 * - SIGMOID: The Sigmoid function \( \sigma(x) = \frac{1}{1 + e^{-x}} \) compresses 
 *   input values to a range between 0 and 1. It is useful for binary classification but can 
 *   suffer from vanishing gradients for large or small inputs.
 * 
 * - SIGMOID_PRIME: The derivative of the sigmoid function \( \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) \).
 * 
 * - RELU: The Rectified Linear Unit (ReLU) function \( f(x) = \max(0, x) \) sets negative
 *   values to 0 and leaves positive values unchanged. ReLU is computationally efficient and 
 *   commonly used in deep learning models, but can suffer from the "dying ReLU" problem where
 *   neurons can become inactive.
 * 
 * - RELU_PRIME: The derivative of the ReLU function \( f'(x) = 1 \) for \( x > 0 \) and \( 0 \) otherwise.
 * 
 * - TANH: The hyperbolic tangent function \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
 *   compresses input values to the range between -1 and 1. Like Sigmoid, it can suffer from
 *   vanishing gradients, but its output is zero-centered, which can be beneficial in training.
 * 
 * - TANH_PRIME: The derivative of the tanh function \( \tanh'(x) = 1 - \tanh^2(x) \).
 * 
 * - SOFTMAX: The Softmax function normalizes a vector of values into a probability 
 *   distribution. It is used in the output layer of neural networks for multi-class classification.
 *   The softmax function ensures that the sum of all output values equals 1.
 * 
 */
enum class ActivationType {
    SIGMOID,        ///< Sigmoid activation function.
    SIGMOID_PRIME,  ///< Derivative of the sigmoid activation function.
    RELU,           ///< Rectified Linear Unit activation function.
    RELU_PRIME,     ///< Derivative of the ReLU activation function.
    TANH,           ///< Hyperbolic tangent activation function.
    TANH_PRIME,     ///< Derivative of the tanh activation function.
    SOFTMAX,        ///< Softmax activation function for multi-class classification.
    INVALID
};


/**
 * @brief Applies the specified activation function to a vector of inputs.
 * 
 * This function takes an activation function (SIGMOID, RELU, TANH, or SOFTMAX) and 
 * applies it element-wise to the input vector. For the softmax activation, it uses 
 * a numerically stable version by adjusting with the maximum input value to prevent 
 * overflow during exponentiation.
 * 
 * - SIGMOID: Applies the sigmoid function \( \sigma(x) = \frac{1}{1 + e^{-x}} \).
 * - RELU: Applies the ReLU function \( f(x) = \max(0, x) \).
 * - TANH: Applies the hyperbolic tangent \( f(x) = \tanh(x) \).
 * - SOFTMAX: Computes the softmax function in a stable manner by subtracting the maximum 
 *   input from each value before exponentiation to avoid numerical issues.
 * 
 * @param activationFunction The type of activation function to apply (SIGMOID, RELU, TANH, SOFTMAX).
 * @param inputs The vector of input values to which the activation function will be applied.
 * 
 * @return A vector of values after the activation function has been applied.
 * 
 * @note The softmax function normalizes the vector of inputs, ensuring the outputs sum to 1.
 */
std::vector<double> Activation(ActivationType activationFunction, std::vector<double> inputs);


/**
 * @brief Converts the ActivationType enum to its corresponding string representation.
 * 
 * This utility function returns a human-readable string corresponding to the specified 
 * activation function. It helps in logging, debugging, or displaying the type of activation 
 * function being used in the model.
 * 
 * @param activationFunction The activation function enum (SIGMOID, RELU, TANH, SOFTMAX).
 * 
 * @return A string representing the activation function name (e.g., "Sigmoid", "ReLU", "Tanh", "Softmax").
 */
std::string ActivationTypeToString(ActivationType activationFunction);


/**
 * @brief Selects the derivate of the activation function based on the activation type.
 * 
 * This function returns the derivative of the specified activation function. It is used
 * during backpropagation to compute the gradient of the loss function with respect to the
 * output of the activation function. The derivative is used to update the weights and biases
 * of the neural network during training.
 * 
 * @param activationFunction The type of activation function for which to compute the derivative.
 * 
 * @return The derivative of the activation function (e.g., SIGMOID_PRIME, RELU_PRIME, TANH_PRIME).
 */
ActivationType select_dActivation(ActivationType activationFunction);


#endif // ACTIVATION_HPP
