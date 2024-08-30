#ifndef IMAGEEXTRACTOR_HPP
#define IMAGEEXTRACTOR_HPP

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "toolkit.hpp"


/**
 * @brief Extracts datasets from CSV files in a specified directory.
 * 
 * This function iterates through all the files in the given directory,
 * identifies CSV files, and processes them to extract images based on the 
 * dataset specified in the CSV files. It reports errors if any issues 
 * arise during the extraction process and informs the user about the 
 * success or failure of the operation.
 * 
 * @param path The path to the directory containing CSV files.
 * 
 * @return 0 on success, 1 on error.
 */
int datasetExtractor(const std::string& path);


/**
 * @brief Imports a dataset from a CSV file and converts it into images.
 * 
 * This function reads pixel data from a CSV file, skipping the first line, 
 * and stores the pixel values in a dataset vector. It then converts this 
 * dataset into images and saves them in the specified output directory. 
 * If the output directory is not empty, it returns early without 
 * processing the CSV file.
 * 
 * @param csvFilePath The path to the input CSV file containing pixel data.
 * @param outputDir The directory where the generated images will be saved.
 * 
 * @return The total number of images converted from the CSV file. Returns 
 *         0 if the output directory is not empty, and -1 if there was an 
 *         error opening the CSV file.
 */
int importCSVDataset(const std::string& csvFilePath, const std::string& outputDir);


/**
 * @brief Converts a dataset of pixel values into images and saves them as PNG files.
 * 
 * This function takes a vector of pixel values representing images and generates 
 * corresponding image files in the specified output directory. Each image is 
 * labeled according to the first value in the pixel vector and is created with 
 * dimensions 28x28 (suitable for MNIST dataset images).
 * 
 * @param dataset A 2D vector containing the dataset where each inner vector 
 *                represents an image with the first element as the label and 
 *                the remaining elements as pixel values.
 * @param outputDir The directory where the generated images will be saved.
 * 
 * @return The total number of images created from the dataset. Returns -1 if 
 *         there was an error saving any image or if no images were created.
 */
int csvToImages(const std::vector<std::vector<int>>& dataset, const std::string& outputDir);


#endif // IMAGEEXTRACTOR_HPP
