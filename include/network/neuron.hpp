#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>


/**
 * @brief Class representing a single neuron in a neural network.
 * 
 * This class encapsulates the behavior of a neuron, including its inputs, weights,
 * bias, and output. The neuron initializes its weights and bias randomly and can
 * calculate its output based on the input values and the weights. It also allows
 * for updating weights and bias.
 */
class Neuron {

    public:

        double bias;                   ///< The bias value of the neuron.
        int inputSize;                 ///< The number of inputs to this neuron.
        std::vector<double> weights;   ///< The weights corresponding to each input.


        /**
         * @brief Constructs a Neuron with the specified input and output sizes.
         * 
         * This constructor initializes the bias and weights for the neuron based on 
         * the given input and output sizes. The bias is initialized to a default value, 
         * while the weights are initialized using the Glorot (Xavier) initialization 
         * method, which helps in maintaining a good variance throughout the layers during 
         * training.
         * 
         * @param inputSize The number of inputs to the neuron.
         * @param outputSize The number of outputs from the neuron.
         */
        Neuron(int inputSize, int outputSize);


        /**
         * @brief Sets the weights of the neuron.
         *  
         * This method updates the weights vector of the neuron.
         * 
         * @param weight the new weights vector.
         */
        void setWeights(std::vector<double> weight);


        /**
         * @brief Sets the bias of the neuron.
         * 
         * This method updates the bias value of the neuron.
         * 
         * @param bias The new bias value.
         */
        void setBias(double bias);


        /**
         * @brief Computes the output of the neuron based on inputs, weights, and bias.
         * 
         * This method calculates the neuron's output using the provided inputs.
         * It sums the product of inputs and weights and adds the bias.
         * 
         * @param inputs A vector of input values.
         * @return The calculated output of the neuron.
         */
        double getOutput(std::vector<double> inputs);



    private:

        static std::default_random_engine re;  ///< Random engine for generating weights and bias.


        /**
         * @brief Initializes the bias for the neuron.
         * 
         * This function sets the bias to a default value. In this implementation, the 
         * bias is initialized to 0.0. This can be modified in the future for different 
         * initialization strategies.
         * 
         * @return The initialized bias value (currently 0.0).
         */
        double initializeBias();


        /**
         * @brief Initializes the weights of the neuron randomly.
         * 
         * This method generates random weights for the inputs of the neuron.
         * 
         * @param inputSize The number of weights to initialize.
         * @return A vector containing the initialized weights.
         */
        std::vector<double> standardInitializeWeights(int inputSize);


        /**
         * @brief Initializes the weights for the neuron.
         * 
         * This function initializes the weights of the neuron using the Glorot (Xavier) 
         * initialization method, which is a common practice in neural network training. 
         * It aims to keep the scale of the gradients roughly the same in all layers, 
         * helping to prevent the vanishing/exploding gradient problem.
         * 
         * @param inputSizeLayer The number of inputs to the layer that this neuron belongs to.
         * @param outputSizeLayer The number of outputs from the layer that this neuron belongs to.
         * 
         * @return A vector of initialized weights for the neuron.
         */
        std::vector<double> initializeWeights(int inputSizeLayer, int outputSizeLayer);

};


#endif // NEURON_HPP
