#ifndef TOOLKIT_HPP
#define TOOLKIT_HPP

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <chrono>
#include <iomanip>  // For std::put_time
#include <sstream>  // For std::ostringstream
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#endif

#include "imageExtractor.hpp"


/**
 * @brief Structure to store image data and its associated label.
 * 
 * This structure is used to hold the pixel data for an image, 
 * the corresponding label (as an integer), and the label represented 
 * as a one-hot encoded vector.
 * 
 * @param imagePixelVector A vector of doubles representing the pixel values of the image.
 *                         This contains the flattened image data (e.g., grayscale values).
 * @param label An integer representing the label or class associated with the image.
 *              For example, in digit classification, this might be the digit (0-9).
 * @param labelVector A one-hot encoded vector of doubles representing the label. 
 *                    This is used in neural network training where the label is represented 
 *                    as a vector with a 1 at the index of the correct class and 0 elsewhere.
 */
struct VectorLabel
{
    std::vector<double> imagePixelVector;
    int label;
    std::vector<double> labelVector;
};


/**
 * @brief Structure to store command-line arguments and dataset paths for training and testing.
 * 
 * This structure is used to store various arguments and paths required for running 
 * training and testing tasks in a machine learning model. It also holds flags to indicate 
 * if the model should be trained or tested, and if pre-existing weights and biases should be used.
 * 
 * @param Train A boolean flag indicating whether to run the training process. Defaults to false.
 * @param Test A boolean flag indicating whether to run the testing process. Defaults to false.
 * @param hasWeightsBiases A boolean flag indicating if the weights and biases file is provided. 
 *                         Defaults to false.
 * @param TrainDatasetPath A string that specifies the path to the training dataset.
 * @param TrainDatasetImages A vector of strings that stores the paths to individual training dataset images.
 * @param TestDatasetPath A string that specifies the path to the testing dataset.
 * @param TestDatasetImages A vector of strings that stores the paths to individual testing dataset images.
 * @param WeightsBiasesPath A string that specifies the path to the file containing the pre-trained weights and biases.
 */
struct Arguments
{
    bool Train = false;
    bool Test = false;
    bool hasWeightsBiases = false;
    std::string TrainDatasetPath = "";
    std::vector<std::string> TrainDatasetImages;
    std::string TestDatasetPath = "";
    std::vector<std::string> TestDatasetImages;
    std::string WeightsBiasesPath = "";
    double learningRate = 0.0;
    int batchSize = 0;
    int epochs = 0;
};


/**
 * @brief Creates a directory if it does not already exist.
 * 
 * This function constructs a folder path by combining a base path and a folder 
 * name. If the specified directory does not exist, it creates the directory 
 * along with any necessary parent directories.
 * 
 * @param basePath The base path where the new folder will be created.
 * @param folderName The name of the folder to create.
 * 
 * @return The full path of the created or existing directory.
 */
std::string makeFolder(const std::string& basePath, const std::string& folderName);


/**
 * @brief Extracts the label from an image file path.
 * 
 * This function extracts the class label from the filename of an image.
 * It assumes that the label is located just before the last underscore ("_")
 * in the file path and is a single digit. The function converts the extracted
 * substring into an integer and returns it as the label.
 * 
 * Example: For the file path "images/sample_3.png", the function would extract 
 * and return the label 3.
 * 
 * @param imagePath The file path of the image as a string.
 * 
 * @return An integer representing the extracted label from the image path.
 * 
 * @note The function assumes that the label is always a single digit just before 
 *       the last underscore ("_") in the file name.
 */
int labelExtractor(const std::string& imagePath);


/**
 * @brief Generates a one-hot encoded vector for a given label.
 * 
 * This function creates a one-hot encoded vector of length 10, where the position 
 * corresponding to the input label is set to 1.0, and all other positions are set 
 * to 0.0. This is typically used in classification tasks where the label corresponds 
 * to a class (e.g., digits 0-9).
 * 
 * Example: If the label is 3, the function will return the vector: 
 * [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
 * 
 * @param label The class label as an integer (expected to be between 0 and 9).
 * 
 * @return A std::vector<double> representing the one-hot encoded label.
 * 
 * @note The input label should be between 0 and 9. If the label is out of this range, 
 *       the function may produce undefined behavior.
 */
std::vector<double> trueLabel(int label);


/**
 * @brief Retrieves a list of image paths from a dataset directory or a single image file.
 * 
 * This function takes the path to a dataset (or image file) and returns a vector of 
 * strings containing the paths to each image. If the input path points to a single image 
 * file with a ".png", ".jpg", or ".jpeg" extension, the function returns a vector containing 
 * just that file path. Otherwise, it treats the path as a directory and returns the paths 
 * to all image files in that directory.
 * 
 * @param datasetPath The path to the dataset directory or image file as a string.
 * 
 * @return A std::vector<std::string> containing the paths to the images in the dataset 
 *         or a single image file path.
 * 
 * @note The function checks for ".png", ".jpg", or ".jpeg" file extensions to determine 
 *       if the input is a single image file. For directories, it retrieves all files without 
 *       filtering for specific image formats.
 */
std::vector<std::string> datasetImagesVector(const std::string& datasetPath);


/**
 * @brief 
 * 
 * @param inputParams 
 * @param argc 
 * @param inputToParse 
 * 
 * @return
 */
int parser(Arguments& inputParams, int argc, char** inputToParse);


/**
 * @brief Converts an image to a pixel vector and extracts its label.
 * 
 * This function reads an image from the given file path, converts the image to grayscale, 
 * and then stores the pixel values as a vector of doubles in the `VectorLabel` structure. 
 * It also extracts the label from the image file name and generates a one-hot encoded 
 * label vector.
 * 
 * The grayscale pixel values are converted to `double` precision and flattened into a 
 * one-dimensional vector, which is stored in `vecLabel.imagePixelVector`. The label 
 * is extracted from the image path and stored in `vecLabel.label`, and a one-hot 
 * encoded vector for the label is generated and stored in `vecLabel.labelVector`.
 * 
 * @param vecLabel Reference to a `VectorLabel` structure that will store the pixel vector, 
 *        extracted label, and one-hot encoded label vector.
 * @param imagePath The file path to the image as a string. The label is extracted from the 
 *        image file name, assuming the label is a single digit just before the last underscore.
 * 
 * @note The function assumes the image is in grayscale format and the label is part of the 
 *       image filename (e.g., "image_0_1.png" where 1 is the label).
 * 
 * @warning The function uses OpenCV to load and process the image, so ensure OpenCV is 
 *          properly configured and linked in your project.
 */
void imageToVectorAndLabel(VectorLabel& vecLabel, std::string imagePath);


/**
 * @brief Return the current date and time as a string.
 * 
 * This function retrieves the current date and time from the system clock and
 * formats it as a string in the format "YYYY-MM-DD HH:MM:SS". The resulting string
 * contains the year, month, day, hour, minute, and second separated by hyphens and colons.
 * 
 * @return A string representing the current date and time in the format "YYYY-MM-DD HH:MM:SS" (MM_DD_YY_HH_mm_ss).
 */
std::string getCurrentDateTime();


/**
 * @brief Return the current date as a string.
 * 
 * This function retrieves the current date from the system clock and
 * formats it as a string in the format "YYYY-MM-DD". The resulting string
 * contains the year, month, and day separated by hyphens.
 * 
 * @return A string representing the current date in the format "YYYY-MM-DD" (MM_DD_YY).
 */
std::string getCurrentDate();


#endif // TOOLKIT_HPP