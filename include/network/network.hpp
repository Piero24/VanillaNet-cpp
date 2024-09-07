#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

#include "layer.hpp"
#include "weightsBiasExtractor.hpp"


/**
 * @brief 
 * 
 */
class Network {

    public:

        std::vector<std::shared_ptr<Layer>> Layers;  ///< A vector containing the layers in the network.
        double loss;                                 ///< The loss value for the network.
        double lossPrime;                            ///< The derivative of the loss function.
        int standardLayerCount = 0;                  ///< The number of standard layers in the network.
        int activationLayerCount = 0;                ///< The number of activation layers in the network.


        /**
         * @brief 
         * 
         */
        Network();


        /**
         * @brief 
         * 
         * @param layer 
         */
        void addLayer(const Layer& layer);


        /**
         * @brief 
         * 
         * @param activationLayer 
         */
        void addLayer(const ActivationLayer& activationLayer);


        /**
         * @brief 
         * 
         * @param weignthsBiases 
         */
        void importWeightsBiases(std::vector<BiasesWeights> weignthsBiases);


        /**
         * @brief 
         * 
         * @return
         */
        std::vector<BiasesWeights> saveWeightsBiases();


        /**
         * @brief 
         * 
         * @param loss 
         * @param lossPrime 
         */
        void setLoss(double loss, double lossPrime);


        /**
         * @brief 
         * 
         * @param inputs 
         * @return 
         */
        std::vector<double> forwardPropagation(const std::vector<double>& inputs);

};


#endif // NETWORK_HPP
