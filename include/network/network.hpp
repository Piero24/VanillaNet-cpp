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


        /**
         * @brief Verifies the correct placement of the Softmax activation layer.
         * 
         * This function checks the sequence of layers in the neural network to ensure 
         * that the Softmax activation function, if present, is only used in the last layer.
         * 
         * - The network is represented by a collection of layers (`Layers`), and this function 
         *   iterates over each layer to find activation layers.
         * - It dynamically casts the current layer to an `ActivationLayer` to access the 
         *   specific activation function used in that layer.
         * - If a Softmax activation layer is found in any position other than the last layer,
         *   the function prints an error message and terminates the program.
         * 
         * The Softmax function is only valid as the final activation in the network, 
         * particularly in classification problems. Having Softmax in an intermediate 
         * layer can lead to incorrect model behavior or misinterpretations of network 
         * outputs.
         * 
         * @note If Softmax is not in the final layer, an error message is displayed, 
         * and the program will exit with status code 1.
         * 
         * @throws Terminates the program if Softmax is not the last activation layer.
         */
        void checkSoftmaxLastLayer();


        /**
         * @brief Sets the loss function for the network and its corresponding derivative.
         * 
         * This function allows you to assign a specific loss function to the neural network.
         * In addition to setting the loss function, it also automatically selects and assigns 
         * the derivative (gradient) of the loss function, which will be used during backpropagation 
         * to compute gradients for weight updates.
         * 
         * - The loss function (`lossFunction`) is set to the one provided as an argument.
         * - The corresponding derivative (`lossFunctionPrime`) is selected using a helper 
         *   function `select_LossFunction_prime()`, which maps the chosen loss function 
         *   to its gradient calculation function.
         * 
         * This design ensures that both the loss function and its gradient are consistently 
         * paired, reducing the chance of errors during training.
         * 
         * @param lossFunction The loss function to be used by the network (e.g., MSE, CrossEntropy).
         * 
         * @note The `lossFunctionPrime` is essential for backpropagation, as it calculates the 
         * gradient of the loss with respect to the output of the network.
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


        /**
         * @brief Computes the loss value based on the chosen loss function.
         * 
         * This function calculates the loss value for the given true labels (`yTrue`) and 
         * predicted values (`yPredicted`) using the specified loss function. 
         * 
         * - The function supports multiple loss functions such as:
         *     - **SQUARED_ERROR**: Computes the sum of squared errors.
         *     - **MEAN_SQUARED_ERROR**: Computes the mean of squared errors.
         *     - **CROSS_ENTROPY**: Computes the binary cross-entropy loss.
         * 
         * After computing the loss, it also calculates and stores the derivative of the loss 
         * function (by calling `lossPrime`) for use during backpropagation.
         * 
         * @param yTrue A vector containing the true labels.
         * @param yPredicted A vector containing the predicted output values from the network.
         * @return The computed loss value as a double.
         * 
         * @note The function stores the computed loss value (`this->lossValue`) and also 
         * prepares the loss gradient (`this->lossPrimeValue`) for use in backpropagation.
         */
        double loss(const std::vector<double>& yTrue, const std::vector<double>& yPredicted);


        /**
         * @brief Computes the derivative (gradient) of the loss function for backpropagation.
         * 
         * This function calculates the gradient of the loss function with respect to 
         * the predicted values (`yPredicted`), which is essential for updating the 
         * network's weights during backpropagation.
         * 
         * - The function supports computing the gradient for multiple loss functions such as:
         *     - **SQUARED_ERROR_PRIME**: Derivative of the sum of squared errors.
         *     - **MEAN_SQUARED_ERROR_PRIME**: Derivative of the mean squared errors.
         *     - **CROSS_ENTROPY_PRIME**: Derivative of the binary cross-entropy loss.
         * 
         * @param yTrue A vector containing the true labels.
         * @param yPredicted A vector containing the predicted output values from the network.
         * @return A vector containing the gradient of the loss function with respect to each output.
         * 
         * @note The computed gradient is stored in `this->lossPrimeValue`, which will be used 
         * during backpropagation to update the weights in the network.
         */
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
