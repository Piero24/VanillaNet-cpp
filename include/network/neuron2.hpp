#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "activation.hpp"

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

        double bias;                         ///< The bias value of the neuron.
        double output;                       ///< The output value of the neuron after activation.
        int numberOfInputs;                 ///< The number of inputs to this neuron.
        std::vector<double> inputs;         ///< The input values to the neuron.
        std::vector<double> weights;        ///< The weights corresponding to each input.


        /**
         * @brief Constructs a Neuron object with the specified inputs.
         * 
         * This constructor initializes the neuron with a set of inputs, randomizes
         * the weights and bias, and calculates the initial output.
         * 
         * @param inputs A vector of input values for the neuron.
         */
        Neuron(std::vector<double> inputs);


        /**
         * @brief Sets the weight at a specified position.
         * 
         * This method updates the weight of the neuron at the specified index.
         * 
         * @param weight The new weight value.
         * @param position The index of the weight to be updated.
         */
        void setWeights(double weight, int position);


        /**
         * @brief Sets the bias of the neuron.
         * 
         * This method updates the bias value of the neuron.
         * 
         * @param weight The new bias value.
         */
        void setBias(double weight);

        
        /**
         * @brief Recalculates the output of the neuron.
         * 
         * This method updates the output value based on the current inputs, 
         * weights, and bias.
         */
        void recalculateOutput();
        
    private:

        static std::default_random_engine re;  ///< Random engine for generating weights and bias.

        
        /**
         * @brief Initializes the weights of the neuron randomly.
         * 
         * This method generates random weights for the inputs of the neuron.
         * 
         * @param numberOfInputs The number of weights to initialize.
         * @return A vector containing the initialized weights.
         */
        std::vector<double> initializeWeights(int numberOfInputs);


        /**
         * @brief Initializes the bias of the neuron randomly.
         * 
         * This method generates a random bias value for the neuron.
         * 
         * @return The initialized bias value.
         */
        double initializeBias();


        /**
         * @brief Computes the output of the neuron based on inputs, weights, and bias.
         * 
         * This method calculates the neuron's output using the provided inputs, weights,
         * and bias. It sums the product of inputs and weights and adds the bias.
         * 
         * @param inputs A vector of input values.
         * @param weights A vector of weight values corresponding to the inputs.
         * @param bias The bias value for the neuron.
         * @return The calculated output of the neuron.
         */
        double getOutput(std::vector<double> inputs, std::vector<double> weights, int bias);

    
};


#endif // NEURON_HPP
