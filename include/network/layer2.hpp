#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>

#include "neuron2.hpp"
#include "activation.hpp"


class Layer {

    public:

        std::vector<Neuron> neurons;                ///< A vector containing the neurons in the layer.
        int numberOfNeurons;                        ///< The number of neurons in the layer.
        std::vector<double> inputs;                 ///< The input values to the layer.
        ActivationType activationFunction;          ///< The activation function to be applied to the layer's outputs.
        std::vector<double> outputs;                ///< The output values of the layer after activation.

        /**
         * @brief Constructs a Layer object with a specified number of neurons and inputs.
         * 
         * This constructor initializes the layer with the given number of neurons and
         * populates the neurons with the provided input values. It also calculates the
         * initial outputs of the layer.
         * 
         * @param numberOfNeurons The number of neurons to be created in the layer.
         * @param inputs A vector of input values for the layer.
         */
        Layer(int numberOfNeurons, std::vector<double> inputs);

        /**
         * @brief Imports weights and biases into the layer's neurons.
         * 
         * This method updates the weights and biases of each neuron in the layer with the
         * provided vectors. It checks for size compatibility between the input weights, biases,
         * and the number of neurons in the layer before proceeding with the updates. If any
         * mismatches are found, appropriate error messages are printed, and the method returns 
         * early without modifying the neurons.
         * 
         * @param weights A 2D vector containing the weights for each neuron, where each inner vector
         *                corresponds to the weights of a specific neuron.
         * @param biases A vector containing the biases for each neuron.
         */
        void importWeightsBiases(std::vector<std::vector<double>> weights, std::vector<double> biases);
    
    private:

        /**
         * @brief Initializes the neurons in the layer.
         * 
         * This method creates a vector of Neuron objects initialized with the given
         * input values.
         * 
         * @param numberOfNeurons The number of neurons to initialize.
         * @param inputs A vector of input values to be assigned to each neuron.
         * @return A vector containing the initialized neurons.
         */
        std::vector<Neuron> initializeNeurons(int numberOfNeurons, std::vector<double> inputs);


        /**
         * @brief Computes the output of the layer based on its neurons.
         * 
         * This method retrieves the outputs from each neuron and returns them as a
         * vector.
         * 
         * @param neurons A vector containing the neurons in the layer.
         * @return A vector of output values corresponding to each neuron.
         */
        std::vector<double> layerOutput(std::vector<Neuron> neurons);

};


#endif // LAYER_HPP
