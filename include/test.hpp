#ifndef TEST_HPP
#define TEST_HPP

#include "network.hpp"
#include "toolkit.hpp"
#include "lossFunctions.hpp"


struct TestResult
{
    int trueValue;
    int predictedValue;
    double loss;
    std::string imagePath;
};


int networkTest(Network &net, Arguments &inputParams);


#endif // TEST_HPP
