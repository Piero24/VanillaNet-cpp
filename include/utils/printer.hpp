#ifndef PRINTER_HPP
#define PRINTER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <iomanip> // For std::setw
#ifdef _WIN32
#include <windows.h>
#endif

#include "imageExtractor.hpp"
#include "network.hpp"
#include "toolkit.hpp"


/**
 * @brief Clears the console screen.
 * 
 * This function clears the console screen, providing a clean 
 * display for output.
 */
void clearScreen();


/**
 * @brief Retrieves the width of the terminal window.
 * 
 * This function checks the operating system and retrieves the current 
 * width of the terminal window in characters. It uses specific APIs 
 * for Windows and Unix/Linux.
 * 
 * @return The width of the terminal in characters.
 */
int getTerminalWidth();


/**
 * @brief Prints a string centered in the terminal with leading/trailing characters.
 * 
 * This function calculates the necessary leading spaces to center the 
 * given text in the terminal. It prints the specified character on 
 * both sides of the text to create a framed effect.
 * 
 * @param text The string to be centered and printed.
 * @param c The character to use for leading and trailing padding.
 */
void printCentered(const std::string &text, char c);


/**
 * @brief Prints a horizontal line across the terminal.
 * 
 * This function prints a horizontal line made up of the specified 
 * character, spanning the entire width of the terminal.
 * 
 * @param c The character to use for the horizontal line.
 */
void printHorizontalLine(char c);


/**
 * @brief Prints information about the neural network and selected parameters.
 * 
 * This function displays the welcome message, selected parameters for 
 * the neural network, and relevant details about the training/testing 
 * datasets and network configuration.
 * 
 * @param inputParams The parameters inputted by the user.
 * @param net The neural network object containing its configuration.
 */
void infoPrinter(Arguments& inputParams, Network& net);


/**
 * @brief Prints the final results after testing/training.
 * 
 * This function displays the final accuracy, loss, and classification 
 * results in a formatted manner for better readability.
 * 
 * @param accuracy The accuracy percentage of the model.
 * @param loss The average loss of the model.
 * @param corrects The number of correctly classified samples.
 * @param total The total number of samples.
 * @param title The title to display above the results.
 */
void finalResultPrinter(double accuracy, double loss, int corrects, int total, std::string title);


#endif // PRINTER_HPP