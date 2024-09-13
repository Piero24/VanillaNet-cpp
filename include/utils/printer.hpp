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


int getTerminalWidth();


void printCentered(const std::string &text, char c);


void printHorizontalLine(char c);


/**
 * @brief 
 * 
 * @param inputParams 
 * 
 * @return
 */
void infoPrinter(Arguments& inputParams, Network& net);

void finalResultPrinter(double accuracy, double loss, int corrects, int total, std::string title);


#endif // PRINTER_HPP