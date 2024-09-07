#include "train.hpp"


int networkTrain(Network &net, Arguments &inputParams, int epochs, int batchSize, double learningRate)
{
    if (!inputParams.Train)
        return 0;

    std::vector<TrainResult> trainResults;

    // Shuffle the training data
    auto rng = std::default_random_engine {};

    for (int i = 0; i < epochs; i++)
    {
        std::shuffle(inputParams.TrainDatasetImages.begin(), inputParams.TrainDatasetImages.end(), rng);
        std::vector<std::vector<std::string>> batches = splitIntoBatches(inputParams.TrainDatasetImages, batchSize);
        int batchCount = 0;

        for(const auto& batch : batches)
        {
            double lossSum = 0.0;

            for (const auto& imagePath : batch)
            {
                TrainResult train;

                VectorLabel vecLabel;
                imageToVectorAndLabel(vecLabel, imagePath);
                train.trueValue = vecLabel.label;
                train.imagePath = imagePath;

                train.epoch = i;
                train.batch = batchCount;

                // forward pass
                std::vector<double> outputOput = net.forwardPropagation(vecLabel.imagePixelVector);

                // calculate loss
                train.loss = mse_loss(vecLabel.labelVector, outputOput);
                net.setLoss(train.loss, 0.0);
                lossSum += train.loss;

                trainResults.push_back(train);
            }
            
            // calculate average loss
            double averageLoss = lossSum / batch.size();

            // backward pass
            // TODO: implement backpropagation                
                
            // update weights and biases
            // TODO: implement weight and bias update

            WeightsBiasesToJSON(net);
            batchCount++;
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