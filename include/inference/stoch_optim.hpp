#pragma once

#include "headers.hpp"

/**
 * @brief Computes adaptive learning rates for each parameter. Has an EWMA of squared gradients.
 */
class RMSProp {
private:
    std::vector<double> _parameters;
    double _variance;
    double _learning_rate;
    double _ewma;
    static double _epsilon;
    unsigned int _t = 1;

public:
    /**
     * @brief RMSProp constructor
     * @param starting_parameters
     * @param starting_variance
     * @param learning_rate
     * @param ewma Exponentially-Weighted Moving Average
     */
    RMSProp(const std::vector<double>& starting_parameters, double starting_variance, double learning_rate,
            double ewma);

    // TODO: copy/move constructor/assignment

    /**
     * @brief
     * @param gradient
     * @return
     */
    std::vector<double> update(double gradient);
};

/**
 * @brief Adaptive Moment Estimation.
 * @brief Computes adaptive learning rates for each parameter.
 * @brief Has an EWMA of past gradients and squared gradients.
 */
class ADAM {
private:
    std::vector<double> _parameters;
    double _f_gradient = 0.0;
    double _variance;
    double _learning_rate;
    double _ewma_1;
    double _ewma_2;
    static double _epsilon;
    int _t = 1;

public:
    /**
     * @brief ADAM constructor
     * @param starting_parameters
     * @param starting_variance
     * @param learning_rate
     * @param ewma1 Exponentially-Weighted Moving Average
     * @param ewma2 Exponentially-Weighted Moving Average
     */
    ADAM(const std::vector<double>& starting_parameters, double starting_variance, double learning_rate, double ewma1,
         double ewma2);

    // TODO: copy/move constructor/assignment

    /**
     * @brief
     * @param gradient
     * @return
     */
    std::vector<double> update(double gradient);
};

double RMSProp::_epsilon = pow(10.0, -8);
double ADAM::_epsilon    = pow(10.0, -8);