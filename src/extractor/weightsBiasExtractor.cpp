#include "weightsBiasExtractor.hpp"


std::vector<BiasesWeights> parseJSON(const std::string& jsonString)
{
    std::vector<BiasesWeights> importedWeightsAndBiases;

    std::ifstream file(jsonString);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return {};
    }

    nlohmann::json data = nlohmann::json::parse(file);
    file.close();

    for (int i = 1; i <= data.size() / 2; i++)
    {
        BiasesWeights bw;
        bw.BiasName = "fc" + std::to_string(i) + ".bias";
        bw.WeightsName = "fc" + std::to_string(i) + ".weight";

        if (data.contains(bw.BiasName) && data.contains(bw.WeightsName)) {
            // Extract bias and weight values
            std::vector<double> bias_vector = data[bw.BiasName].get<std::vector<double>>();
            std::vector<std::vector<double>> weight_matrix = data[bw.WeightsName].get<std::vector<std::vector<double>>>();

            // Assign the extracted values to the structure fields
            bw.biases = bias_vector;
            bw.weights = weight_matrix;
        }
        else {
            std::cerr << "The key does not exist." << std::endl;
            return {};
        }

        // Add the BiasesWeights instance to the vector
        importedWeightsAndBiases.push_back(bw);
    }

    return importedWeightsAndBiases;
}


void jsonValuePrinter(const std::vector<BiasesWeights>& importedWeightsAndBiases)
{
    for (const auto& bw : importedWeightsAndBiases)
    {
        printf("Bias name: %s Biases size: %ld\n" , bw.BiasName.c_str(), bw.biases.size());
        for (int i = 0; i < bw.biases.size(); i++)
        {
            printf("Bias %d: %f\n", i+1, bw.biases[i]);
        }
        printf("\n");
        
        printf("Weights name: %s Weights size: %ld\n", bw.WeightsName.c_str(), bw.weights.size());
        for (int i = 0; i < bw.weights.size(); i++)
        {
            printf("Row %d: %ld\n", i+1, bw.weights[i].size());
            for (int j = 0; j < bw.weights[i].size(); j++)
            {
                printf("Weight %d: %.30f\n", j+1, bw.weights[i][j]);
            }
            
        }
        printf("\n");
    }
}


void weightsBiasExtractor(Arguments &inputParams, std::vector<BiasesWeights> &importedWeightsAndBiases)
{
    if (inputParams.hasWeightsBiases)
    {
        importedWeightsAndBiases = parseJSON("./Resources/output/weights/test.json");
        // importedWeightsAndBiases = parseJSON("./Resources/output/weights/mnist_fc128_relu_fc10_log_softmax_weights_biases.json");
        // importedWeightsAndBiases = parseJSON("./Resources/output/weights/mnist_fc128_relu_fc10_sigmoid_weights_biases.json");
        // jsonValuePrinter(importedWeightsAndBiases);
    }
}


nlohmann::json serializeWeightsBiases(const std::vector<BiasesWeights>& savedWB)
{
    nlohmann::json jsonObject;

    for (const auto& wb : savedWB)
    {
        jsonObject[wb.WeightsName] = wb.weights;
        jsonObject[wb.BiasName] = wb.biases;
    }

    return jsonObject;
}


void writeJsonToFile(const nlohmann::json& jsonObject, const std::string& filePath)
{
    std::ofstream file(filePath);
    
    if (file.is_open())
    {
        file << jsonObject.dump(4);  // Pretty print JSON with an indentation of 4 spaces
        file.close();
        std::cout << "JSON written to " << filePath << std::endl;
    } else
    {
        std::cerr << "Unable to open file " << filePath << std::endl;
    }
}
