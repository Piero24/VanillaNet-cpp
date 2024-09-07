#include "saveToJson.hpp"


std::string WeightsBiasesToJSON(Network& net)
{
    std::vector<BiasesWeights> savedWB = net.saveWeightsBiases();
    nlohmann::json jsonWeightsBiases = serializeWeightsBiases(savedWB);
    //std::cout << jsonWeightsBiases.dump(4) << std::endl;

    std::string filePath = "./Resources/output/weights/";
    makeFolder("./Resources", "output");
    makeFolder("./Resources/output", "weights");

    std::string currentDateTime = getCurrentDateTime();
    std::string fileName;

    for (int i = 0; i < net.Layers.size(); i++)
    {
        // Identify the type of layer using the polymorphic method getType
        if (net.Layers[i]->getType() == LayerType::StandardLayer)
        {
            fileName = fileName + "fc" + std::to_string(net.Layers[i]->neurons.size()) + "_";
        }
        else if (net.Layers[i]->getType() == LayerType::ActivationLayer)
        {
            // Use dynamic_cast to access the ActivationLayer
            ActivationLayer* activationLayer = dynamic_cast<ActivationLayer*>(net.Layers[i].get());
            if (activationLayer != nullptr)
            {
                fileName += ActivationTypeToString(activationLayer->activationFunction) + "_";
            }
        }
    }

    filePath = filePath + "/" + fileName + currentDateTime + ".json";
    writeJsonToFile(jsonWeightsBiases, filePath);
    return filePath;
}