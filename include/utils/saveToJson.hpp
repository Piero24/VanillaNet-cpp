#ifndef SAVETOJSON_HPP
#define SAVETOJSON_HPP

class Network; // Forward declaration of Network class

#include <nlohmann/json.hpp>

#include "layer.hpp"
#include "network.hpp"
#include "toolkit.hpp"


/**
 * @brief Saves the weights and biases of the network to a JSON file.
 * 
 * This function takes a Network object and saves the weights and biases of each layer
 * to a JSON file. The weights and biases are stored in a vector of BiasesWeights structures
 * which contain the name of the layer, the biases, and the weights.
 * 
 * @param net The Network object to save the weights and biases from.
 * 
 * @return The path to the JSON file where the weights and biases are saved.
 */
std::string WeightsBiasesToJSON(Network& net);

#endif // SAVETOJSON_HPP