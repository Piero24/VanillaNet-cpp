#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

#include <iostream>
#include <vector>


/**
 * @brief Computes the Mean Squared Error (MSE) loss between two vectors.
 * 
 * The Mean Squared Error (MSE) is a common loss function used in regression tasks.
 * It measures the average of the squares of the differences between the true values 
 * (yTrue) and the predicted values (yPredicted). A lower MSE indicates that the predicted
 * values are closer to the true values.
 * 
 ** Formula: MSE = (1/n) * Σ (yTrue[i] - yPredicted[i])^2
 * 
 * @param yTrue A vector of true/target values. This represents the actual values from the dataset.
 * @param yPredicted A vector of predicted values. These are the values predicted by the model.
 * 
 * @return The Mean Squared Error (MSE) as a double value.
 * 
 * @note Both vectors should have the same length. If they differ in size, this function may lead to 
 *       undefined behavior or errors.
 */
double mse_loss(std::vector<double> yTrue, std::vector<double> yPredicted);


#endif // LOSSFUNCTIONS_HPP
