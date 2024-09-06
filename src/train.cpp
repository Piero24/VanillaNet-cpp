#include "train.hpp"


int networkTrain(Arguments &inputParams, int epochs, int batchSize, double learningRate)
{
    if (!inputParams.Train)
        return 0;

    // Shuffle the training data
    auto rng = std::default_random_engine {};

    for (int i = 0; i < epochs; i++)
    {
        std::shuffle(inputParams.TrainDatasetImages.begin(), inputParams.TrainDatasetImages.end(), rng);
        std::vector<std::vector<std::string>> batches = splitIntoBatches(inputParams.TrainDatasetImages, batchSize);

        for(const auto& batch : batches)
        {
            for (const auto& imagePath : batch)
            {
                // Load the image and label
                VectorLabel vecLabel;
                imageToVectorAndLabel(vecLabel, imagePath);

                // forward pass
                // calculate loss
                // backward pass
                // update weights and biases

            }
        }
    }

    return 0;
}


std::vector<std::vector<std::string>> splitIntoBatches(const std::vector<std::string>& inputVec, int batchSize)
{
    std::vector<std::vector<std::string>> batches;
    int totalSize = inputVec.size();

    for (int i = 0; i < totalSize; i += batchSize)
    {
        // Define the end iterator, making sure not to go past the end of the input vector
        int end = std::min(i + batchSize, totalSize);
        // Create a new batch from the current position (i) to the end
        std::vector<std::string> batch(inputVec.begin() + i, inputVec.begin() + end);
        // Add the batch to the list of batches
        batches.push_back(batch);
    }

    return batches;
}