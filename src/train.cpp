#include "train.hpp"

int networkTrain(Network &net, Arguments &inputParams)
{
    if (!inputParams.Train)
        return 0;

    // Shuffle the training data
    auto rng = std::default_random_engine {};
    
    int totCorrect = 0;
    double totalLoss = 0.0;

    for (int i = 0; i < inputParams.epochs; i++)
    {
        std::shuffle(inputParams.TrainDatasetImages.begin(), inputParams.TrainDatasetImages.end(), rng);
        std::vector<std::vector<std::string>> batches = splitIntoBatches(inputParams.TrainDatasetImages, inputParams.batchSize);

        double epochLossSum = 0.0;
        int epochCorrectImagesCount = 0;

        for(size_t m = 0; m < batches.size(); m++)
        {
            double batchLossSum = 0.0;
            int batchCorrectImagesCount = 0;

            double geometricMeanLoss = 1.0;

            std::vector<std::vector<BiasesWeights>> accumulatedGrad;
            
            for (size_t n = 0; n < batches[m].size(); n++)
            {
                VectorLabel vecLabel;
                imageToVectorAndLabel(vecLabel, batches[m][n]);

                std::vector<double> outputOput = net.forwardPropagation(vecLabel.imagePixelVector);

                // calculate loss
                double lossValue = net.loss(vecLabel.labelVector, outputOput);
                batchLossSum += lossValue;
                geometricMeanLoss *= lossValue; // Multiply the losses

                // backward pass
                std::vector<BiasesWeights> gradWeightsBiases = net.backwardPropagation(net.lossPrimeValue);
                accumulatedGrad.push_back(gradWeightsBiases);

                auto max_element_iter = std::max_element(outputOput.begin(), outputOput.end());

                int predictedLabel = 0;
                if (max_element_iter != outputOput.end())
                    predictedLabel = std::distance(outputOput.begin(), max_element_iter);

                batchCorrectImagesCount += (vecLabel.label == predictedLabel);

                // std::ostringstream ossAcc;
                // ossAcc << std::fixed << std::setprecision(2) << 100.0 * ((double)batchCorrectImagesCount / batches[m].size());

                // std::cout << ">>>> Epoch: " << i+1 << "/" << inputParams.epochs;
                // std::cout << "     Batch: " << m+1 << "/" << batches.size();
                // std::cout << "     Sample: " << n+1 << "/" << batches[m].size();
                // std::cout << "     Loss: " << lossValue;
                // std::cout << "     Batch Accuracy: " << ossAcc.str();
                // std::cout << "%     Predicted: " << predictedLabel;
                // std::cout << "     True: " << vecLabel.label << "\n" << std::endl;
            }

            epochLossSum += batchLossSum;
            epochCorrectImagesCount += batchCorrectImagesCount;

            // calculate average loss
            // double averageLoss = batchLossSum / batches[m].size();
            double batchAccuracy = 100.0 * ((double)batchCorrectImagesCount / batches[m].size());
            geometricMeanLoss = std::pow(geometricMeanLoss, 1.0 / batches[m].size());

            std::ostringstream ossAcc;
            ossAcc << std::fixed << std::setprecision(2) << batchAccuracy;

            std::cout << ">>> Epoch: " << i+1 << "/" << inputParams.epochs;
            std::cout << "     Batch: " << m+1 << "/" << batches.size();
            std::cout << "     Average Loss: " << geometricMeanLoss;
            std::cout << "     Batch Accuracy: " << ossAcc.str();
            std::cout << "%     Predicted Correctly: " << batchCorrectImagesCount << "/" << batches[m].size() << "\n" << std::endl;

            // update weights and biases
            net.updateWeightsBiases(accumulatedGrad, inputParams.learningRate);

            std::string jsonPath = WeightsBiasesToJSON(net);
            // printf(">> Weights and biases saved to: %s\n\n", jsonPath.c_str());
        }

        totalLoss += epochLossSum;
        totCorrect += epochCorrectImagesCount;

        double averageLoss = epochLossSum / inputParams.TrainDatasetImages.size();
        double batchAccuracy = 100.0 * ((double)epochCorrectImagesCount / inputParams.TrainDatasetImages.size());

        std::ostringstream ossAcc;
        ossAcc << std::fixed << std::setprecision(2) << batchAccuracy;

        std::cout << ">> Epoch: " << i+1 << "/" << inputParams.epochs;
        std::cout << "     Average Loss: " << averageLoss;
        std::cout << "     Accuracy: " << ossAcc.str();
        std::cout << "%     Predicted Correctly: " << epochCorrectImagesCount << "/" << inputParams.TrainDatasetImages.size() << "\n" << std::endl;
    }

    std::string title = " TRAINING RESULTS ";
    double lossToPrint = (totalLoss / inputParams.TrainDatasetImages.size()*inputParams.epochs);
    double acc = 100.0 * ((double)totCorrect / (inputParams.TrainDatasetImages.size()*inputParams.epochs));

    finalResultPrinter(acc, lossToPrint, totCorrect, inputParams.TrainDatasetImages.size()*inputParams.epochs, title);

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