#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "activation.hpp"


class Neuron {


    public:

        //
        double bias;
        //
        double output;
        //
        int numberOfInputs;
        //
        std::vector<double> inputs;
        //
        std::vector<double> weights;


        /**
         * @brief 
         * 
         * @param inputs
         * @param activationFunction
         * 
         * @return
         */
        Neuron(std::vector<double> inputs);


        /**
         * @brief 
         * 
         * @param weight
         * @param position
         * 
         * @return
         */
        void setWeights(double weight, int position);


        /**
         * @brief 
         * 
         * @param weight
         * 
         * @return
         */
        void setBias(double weight);

    
    private:

        static std::default_random_engine re;

        
        /**
         * @brief 
         * 
         * @param numberOfInputs
         * 
         * @return
         */
        std::vector<double> initializeWeights(int numberOfInputs);


        /**
         * @brief 
         * 
         * @return
         */
        double initializeBias();


        /**
         * @brief 
         * 
         * @param inputs
         * @param weights
         * @param bias
         * 
         * @return
         */
        double getOutput(std::vector<double> inputs, std::vector<double> weights, int bias);
    
};


#endif // NEURON_HPP
