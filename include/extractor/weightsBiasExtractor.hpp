#ifndef WEIGHTSBIASEXTRACTOR_HPP
#define WEIGHTSBIASEXTRACTOR_HPP

#include <iostream>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "toolkit.hpp"


/**
 * @brief Structure to hold the biases and weights for a specific layer of a neural network.
 * 
 * This structure is used to store the name, biases, and weights of a fully connected (fc) layer 
 * in a neural network. Each layer has an associated bias vector and a weight matrix. These 
 * parameters are typically learned during training and are later extracted from a JSON file.
 */
struct BiasesWeights
{
    int LayerIndex = 0;                     ///< The index of the layer in the network (default: 0).
    std::string BiasName = "";              ///< The name identifier for the bias of this layer.
    std::vector<double> biases;             ///< The vector of bias values for this layer.
    std::string WeightsName = "";           ///< The name identifier for the weights of this layer.
    std::vector<std::vector<double>> weights; ///< The matrix of weight values for this layer.
};


/**
 * @brief Parses a JSON file to extract biases and weights for multiple layers.
 * 
 * This function reads a JSON file containing the biases and weights for a neural network's 
 * layers and stores them in a vector of `BiasesWeights` structures. The file is expected to 
 * contain bias and weight data for fully connected (fc) layers in a neural network, with keys 
 * in the format "fcN.bias" and "fcN.weight" where N is the layer index.
 * 
 * @param jsonString The path to the JSON file containing the neural network parameters.
 * @return std::vector<BiasesWeights> A vector containing the biases and weights for each layer. 
 *                                    Returns an empty vector if the file cannot be opened or if 
 *                                    any required keys are missing.
 */
std::vector<BiasesWeights> parseJSON(const std::string& jsonString);


/**
 * @brief Prints the biases and weights from a vector of BiasesWeights to the console.
 * 
 * This function iterates over a vector of `BiasesWeights` structures, printing the size 
 * and values of biases and weights for each layer to the console. The weights are printed 
 * with high precision (30 decimal places).
 * 
 * @param importedWeightsAndBiases The vector containing biases and weights for each layer.
 */
void jsonValuePrinter(const std::vector<BiasesWeights>& importedWeightsAndBiases);


/**
 * @brief Extracts the biases and weights from a JSON file and stores them in a vector.
 * 
 * This function reads a JSON file containing the biases and weights for a neural network's 
 * layers and stores them in a vector of `BiasesWeights` structures. The file is expected to 
 * contain bias and weight data for fully connected (fc) layers in a neural network, with keys 
 * in the format "fcN.bias" and "fcN.weight" where N is the layer index.
 * 
 * @param inputParams The input parameters for the program.
 * @param importedWeightsAndBiases The vector to store the biases and weights for each layer.
 */
void weightsBiasExtractor(Arguments &inputParams, std::vector<BiasesWeights> &importedWeightsAndBiases);


/**
 * @brief Serializes the biases and weights from a vector of BiasesWeights to a JSON object.
 * 
 * This function converts a vector of `BiasesWeights` structures to a JSON object, storing the 
 * biases and weights for each layer. The JSON object is then returned for further processing 
 * or saving to a file.
 * 
 * @param savedWB The vector containing biases and weights for each layer.
 * @return nlohmann::json The JSON object containing the biases and weights for each layer.
 */
nlohmann::json serializeWeightsBiases(const std::vector<BiasesWeights>& savedWB);


/**
 * @brief Writes a JSON object to a file.
 * 
 * This function writes a JSON object to a file specified by the file path. The JSON object 
 * is formatted with indentation for readability.
 * 
 * @param jsonObject The JSON object to write to the file.
 * @param filePath The path to the file where the JSON object will be saved.
 */
void writeJsonToFile(const nlohmann::json& jsonObject, const std::string& filePath);


#endif // WEIGHTSBIASEXTRACTOR_HPP
