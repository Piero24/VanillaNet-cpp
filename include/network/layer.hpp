#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>

#include "neuron.hpp"
#include "activation.hpp"


/**
 * @brief Enum representing the type of a layer in a neural network.
 * 
 * The LayerType enum is used to specify the type of a layer in a neural network.
 * The two possible types are StandardLayer and ActivationLayer.
 */
enum class LayerType {
    StandardLayer,
    ActivationLayer
};


/**
 * @brief Class representing a layer of neurons in a neural network.
 * 
 * This class encapsulates the behavior of a layer of neurons, including the number of
 * neurons, the number of inputs to each neuron, and the weights and biases of each neuron.
 * It provides methods for computing the output of the layer based on its neurons and for
 * importing weights and biases into the layer.
 */
class Layer {

    public:

        std::vector<Neuron> neurons;                ///< A vector containing the neurons in the layer.
        int inputSize;                              ///< The number of inputs to each neuron in the layer.
        int outputSize;                             ///< The number of neurons in the layer.
        std::vector<double> inputs;                 ///< The input of the layer.
        std::vector<double> outputs;                ///< The output of the layer.


        /**
         * @brief Constructs a Layer object with a specified number of neurons.
         * 
         * This constructor initializes the layer with the given number of neurons (outputSize) and
         * populates the neurons with the provided input values (inputSize).
         * 
         * @param inputSize The number of inputs to each neuron in the layer.
         * @param outputSize The number of neurons to be created in the layer.
         */
        Layer(int inputSize, int outputSize);


        /**
         * @brief Returns the type of the layer.
         * 
         * This method returns the type of the layer as LayerType::StandardLayer.
         * 
         * @return The type of the layer.
         */
        virtual LayerType getType() const { return LayerType::StandardLayer; }


        /**
         * @brief Returns the number of neurons in the layer.
         * 
         * @return The number of neurons in the layer.
         */
        int getLayerSize();


        /**
         * @brief Imports weights and biases into the layer's neurons.
         * 
         * This method updates the weights and biases of each neuron in the layer with the
         * provided vectors. It checks for size compatibility between the input weights, biases,
         * and the number of neurons in the layer before proceeding with the updates. If any
         * mismatches are found, appropriate error messages are printed, and the method returns 
         * early without modifying the neurons.
         * 
         * @param weights A 2D vector containing the weights for each neuron.
         * @param biases A vector containing the biases for each neuron.
         */
        void importWeightsBiases(std::vector<std::vector<double>> weights, std::vector<double> biases);


        /**
         * @brief Saves the weights and biases of the layer's neurons.
         * 
         * This method retrieves the weights and biases of each neuron in the layer and stores
         * them in the provided vectors. It appends the weights and biases of each neuron to the
         * respective vectors.
         * 
         * @param weights A 2D vector to store the weights of each neuron.
         * @param biases A vector to store the biases of each neuron.
         */
        void saveWeightsBiases(std::vector<std::vector<double>>& weights, std::vector<double>& biases);


        /**
         * @brief Computes the output of the layer based on its neurons.
         * 
         * This method retrieves the outputs from each neuron and returns them as a
         * vector.
         * 
         * @param inputs A vector of input values to the layer.
         * @return A vector of output values corresponding to each neuron.
         */
        virtual std::vector<double> forwardPass(std::vector<double> inputs);


        /**
         * @brief Performs the backward pass for a single layer, computing the gradients for the weights, biases, 
         *        and propagating the error back to the previous layer.
         * 
         * @param error The error from the output layer or the next layer (depending on the position of the current layer).
         * @param weights A reference to the weights of the current layer, which will be updated with the computed gradients.
         * @param biases A reference to the biases of the current layer, which will be updated with the computed gradients.
         * @return std::vector<double> The error propagated back to the previous layer.
         */
        virtual std::vector<double> backwardPass(std::vector<double>& error, std::vector<std::vector<double>>& weights, std::vector<double>& biases);

        /**
         * @brief Updates the weights and biases of the neurons in the layer using the calculated gradients and the learning rate.
         * 
         * @param learningRate The step size used for updating the weights and biases.
         * @param gradientsWeights The gradients for the weights of each neuron in the layer.
         * @param gradientsBiases The gradients for the biases of each neuron in the layer.
         */
        void updateWeightsBiases(double learningRate, std::vector<std::vector<double>> weights, std::vector<double> biases);


        /**
         * @brief Destructor for the Layer class.
         */
        virtual ~Layer() { }
    

    private:

        /**
         * @brief Initializes the neurons in the layer.
         * 
         * This method creates a vector of Neuron objects of size outputSize where each neuron
         * is initialized with inputSize. The weights and bias of each neuron are initialized
         * randomly.
         * 
         * @param inputSize The number of inputs to each neuron.
         * @param outputSize The number of neurons to initialize.
         * @return A vector containing the initialized neurons.
         */
        std::vector<Neuron> initializeNeurons(int inputSize, int outputSize);

};


/**
 * @brief Class representing an activation layer in a neural network.
 * 
 * This class extends the Layer class and adds an activation function to the layer.
 * It provides a method for applying the activation function to the layer's outputs.
 * The activation function is applied element-wise to the output values of the layer.
 * The ActivationLayer class is used to introduce non-linearity into the network.
 * The activation function can be specified when creating an ActivationLayer object.
 */
class ActivationLayer : public Layer {

    public:

        ActivationType activationFunction;         ///< The activation function to be applied to the layer's outputs.


        /**
         * @brief Constructs an ActivationLayer object with a specified activation function.
         * 
         * This constructor initializes the activation layer with the given activation function.
         * 
         * @param activationFunction The activation function to be applied to the layer's outputs.
         */
        ActivationLayer(ActivationType activationFunction);


        /**
         * @brief Returns the type of the layer.
         * 
         * This method returns the type of the layer as LayerType::ActivationLayer.
         * 
         * @return The type of the layer.
         */
        virtual LayerType getType() const override { return LayerType::ActivationLayer; }


        /**
         * @brief Applies the activation function to the layer's outputs.
         * 
         * @param inputs A vector of output from the previous layer.
         * @return A vector of output values after applying the activation function.
         */
        std::vector<double> forwardPass(std::vector<double> inputs) override;


        /**
         * @brief Performs the backward pass through the activation layer, applying the derivative of the activation function 
         *        to the error from the next layer.
         * 
         * @param error The error from the next layer that needs to be adjusted based on the activation function.
         * @param weights Not used in this layer, but passed for compatibility with other layers.
         * @param biases Not used in this layer, but passed for compatibility with other layers.
         * @return std::vector<double> The modified error after applying the derivative of the activation function.
         */
        std::vector<double> backwardPass(std::vector<double>& error, std::vector<std::vector<double>>& weights, std::vector<double>& biases) override;

        /**
         * @brief Destructor for the ActivationLayer class.
         */
        virtual ~ActivationLayer() { }

};


#endif // LAYER_HPP
