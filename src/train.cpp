#include "train.hpp"


int networkTrain(Network &net, Arguments &inputParams)
{
    if (!inputParams.Train)
        return 0;

    std::vector<TrainResult> trainResults;

    // Shuffle the training data
    auto rng = std::default_random_engine {};
    int totCorrect = 0;
    double totalLoss = 0.0;

    for (int i = 0; i < inputParams.epochs; i++)
    {
        std::shuffle(inputParams.TrainDatasetImages.begin(), inputParams.TrainDatasetImages.end(), rng);
        std::vector<std::vector<std::string>> batches = splitIntoBatches(inputParams.TrainDatasetImages, inputParams.batchSize);
        int batchCount = 0;
        double epochSumLoss = 0.0;
        int epochCorrect = 0;

        for(const auto& batch : batches)
        {
            double lossSum = 0.0;
            int imageCount = 0;
            int batchCorrect = 0;

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
                totalLoss += train.loss;
                epochSumLoss += train.loss;

                auto max_element_iter = std::max_element(outputOput.begin(), outputOput.end());

                if (max_element_iter != outputOput.end())
                    train.predictedValue = std::distance(outputOput.begin(), max_element_iter);

                totCorrect += train.trueValue == train.predictedValue;
                batchCorrect += train.trueValue == train.predictedValue;
                epochCorrect += train.trueValue == train.predictedValue;

                // printf(">>>> Epoch: %d/%d     Batch: %d/%ld     Sample: %d/%ld     Loss: %.6f     Batch Accuracy: %.2f%%     Predicted: %d     True: %d\n\n", i+1, inputParams.epochs, batchCount+1, batches.size(), imageCount+1, batch.size(), train.loss, 100.0 * ((double)batchCorrect / batch.size()), train.predictedValue, train.trueValue);

                imageCount++;
                trainResults.push_back(train);
            }

            // calculate average loss
            double averageLoss = lossSum / batch.size();

            printf(">>> Epoch: %d/%d     Batch: %d/%ld     Average Loss: %.6f     Batch Accuracy: %.2f%%     Correctly Predicted: %d/%ld\n\n", i+1, inputParams.epochs, batchCount+1, batches.size(), averageLoss, 100.0 * ((double)batchCorrect / batch.size()), batchCorrect, batch.size());
            

            // backward pass
            // TODO: implement backpropagation
                
            // update weights and biases
            // TODO: implement weight and bias update

            std::string jsonPath = WeightsBiasesToJSON(net);
            // printf(">> Weights and biases saved to: %s\n\n", jsonPath.c_str());
            batchCount++;
        }


        printf(">> Epoch: %d/%d     Average Loss: %.6f     Accuracy: %.2f%%     Correctly Predicted: %d/%ld\n\n", i+1, inputParams.epochs, ((double)epochSumLoss / inputParams.TrainDatasetImages.size()), 100.0 * ((double)epochCorrect / inputParams.TrainDatasetImages.size()), epochCorrect, inputParams.TrainDatasetImages.size());

    }

    printf("\n");
    printCentered(" TRAINING RESULTS ", '*');
    printf("\nCorrectly Classified:  %d/%ld\t\t\t\t\t\t\t\t\t\t", totCorrect, inputParams.TrainDatasetImages.size()*inputParams.epochs);
    printf("Accuracy:  %.2f%%\t\t\t\t\t\t\t\t\t\t", 100.0 * ((double)totCorrect / inputParams.TrainDatasetImages.size()*inputParams.epochs));
    printf("Average loss:  %.5f\t\t", (totalLoss / inputParams.TrainDatasetImages.size()*inputParams.epochs));

    printf("\n\n");
    printHorizontalLine('*');

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