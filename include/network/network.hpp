#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

#include "layer.hpp"
#include "weightsBiasExtractor.hpp"
#include "activation.hpp"
#include "lossFunctions.hpp"


/**
 * @brief Represents a neural network composed of multiple layers.
 * 
 * The Network class encapsulates the functionality to create, manage, and 
 * train a neural network. It includes methods for adding layers, performing 
 * forward and backward propagation, and saving/loading weights and biases.
 */
class Network {

    public:

        std::vector<std::shared_ptr<Layer>> Layers;  ///< A vector containing the layers in the network.
        double lossValue;                            ///< The loss value for the network.
        std::vector<double> lossPrimeValue;          ///< The derivative of the loss function.
        int standardLayerCount = 0;                  ///< The number of standard layers in the network.
        int activationLayerCount = 0;                ///< The number of activation layers in the network.
        std::vector<double> inputs;                  ///< The input to the network.
        std::vector<double> output;                  ///< The output of the network.
        LossFunction lossFunction;                   ///< The loss function used by the network.
        LossFunctionPrime lossFunctionPrime;         ///< The derivative of the loss function used by the network.


        /**
         * @brief Constructs a new Network instance.
         * 
         * This constructor initializes the loss and lossPrime to zero and 
         * sets up an empty vector of layers. It prepares the network for 
         * subsequent layer addition and training.
         */
        Network();


        /**
         * @brief Adds a standard layer to the network.
         * 
         * This function takes a Layer object and adds it to the network's 
         * layers. It also increments the count of standard layers.
         * 
         * @param layer The Layer object to be added to the network.
         */
        void addLayer(const Layer& layer);


        /**
         * @brief Adds an activation layer to the network.
         * 
         * This function takes an ActivationLayer object and adds it to the 
         * network's layers. It increments the count of activation layers.
         * 
         * @param activationLayer The ActivationLayer object to be added to the network.
         */
        void addLayer(const ActivationLayer& activationLayer);


        void checkSoftmaxLastLayer();


        /**
         * @brief 
         */
        void addLossFunction(LossFunction lossFunction);


        /**
         * @brief Imports weights and biases into the network.
         * 
         * This function populates the weights and biases of the layers in the 
         * network from the provided vector of BiasesWeights. It ensures that 
         * the number of layers matches the number of weight and bias sets 
         * provided.
         * 
         * @param weightsBiases A vector containing the weights and biases for each layer.
         */
        void importWeightsBiases(std::vector<BiasesWeights> weignthsBiases);


        /**
         * @brief Saves the current weights and biases of the network.
         * 
         * This function collects the weights and biases from each layer and 
         * returns them as a vector of BiasesWeights, which can be used for 
         * saving to a file or for later retrieval.
         * 
         * @return A vector of BiasesWeights containing the weights and biases of the network layers.
         */
        std::vector<BiasesWeights> saveWeightsBiases();


        double loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


        std::vector<double> lossPrime(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


        /**
         * @brief Sets the loss and its derivative for the network.
         * 
         * This function updates the loss and lossPrime attributes of the network, 
         * which are used during training to evaluate performance and guide updates.
         * 
         * @param loss The current loss value.
         * @param lossPrime The derivative of the loss function.
         */
        void setLoss(double loss, const std::vector<double>& lossPrime);


        /**
         * @brief Performs forward propagation through the network.
         * 
         * This function takes an input vector, processes it through each layer 
         * in the network, and returns the final output vector. It updates the 
         * inputs and outputs of the network.
         * 
         * @param inputs A vector containing the input values for the network.
         * @return A vector containing the output values after forward propagation.
         */
        std::vector<double> forwardPropagation(const std::vector<double>& inputs);


        /**
         * @brief Performs backward propagation through the network.
         * 
         * This function calculates the gradients for each layer based on the 
         * output error and returns them in a vector of BiasesWeights. This is 
         * used to adjust the weights and biases during training.
         * 
         * @param outputError A vector containing the error at the output layer.
         * @return A vector of BiasesWeights containing the gradients for each layer.
         * 
         * @note: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
         */
        std::vector<BiasesWeights> backwardPropagation(const std::vector<double>& outputError);


        /**
         * @brief Updates the weights and biases of the network.
         * 
         * This function applies the calculated gradients to the weights and biases 
         * of each layer using the specified learning rate. It utilizes the 
         * average gradients calculated from the accumulated gradients over 
         * multiple batches.
         * 
         * @param accumulatedGrad A vector of accumulated gradients from multiple batches.
         * @param learningRate The rate at which to update the weights and biases.
         */
        void updateWeightsBiases(const std::vector<std::vector<BiasesWeights>>& accumulatedGrad, double learningRate);
    

    private:

        /**
         * @brief Calculates the average gradients from the accumulated gradients.
         * 
         * This function takes a vector of accumulated gradients and computes 
         * the average for each layer's weights and biases. This is used to 
         * ensure stable updates during training.
         * 
         * @param accumulatedGrad A vector of vectors containing accumulated gradients.
         * @return A vector of BiasesWeights containing the average gradients.
         */
        std::vector<BiasesWeights> calculateAverageGradients(const std::vector<std::vector<BiasesWeights>>& accumulatedGrad);
};


#endif // NETWORK_HPP
