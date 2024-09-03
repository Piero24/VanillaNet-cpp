#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

#include <iostream>
#include <vector>


/**
 * @brief 
 * 
 * @param yTrue 
 * @param yPredicted 
 * 
 * @return
 */
double mse_loss(std::vector<double> yTrue, std::vector<double> yPredicted);


#endif // LOSSFUNCTIONS_HPP
