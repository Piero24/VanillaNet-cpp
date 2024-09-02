#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>

#include "neuron.hpp"
#include "activation.hpp"


class Layer {

    public:

        //
        std::vector<Neuron> neurons;
        
        //
        int numberOfNeurons;
        
        //
        std::vector<double> inputs;
        
        //
        ActivationType activationFunction;

        //
        std::vector<double> outputs;
    

        Layer(int numberOfNeurons, std::vector<double> inputs);
    
    private:

        std::vector<Neuron> initializeNeurons(int numberOfNeurons, std::vector<double> inputs);

        std::vector<double> layerOutput(std::vector<Neuron> neurons);

};


#endif // LAYER_HPP
