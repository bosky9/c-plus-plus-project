#pragma once

#include "arima/arima.hpp"

#include <Eigen/Core>
#include <functional>
#include <iostream>

/**
 * @brief Used to have a function and a way to compute its derivative in the same class.
 *
 * @details     Necessary for the LBFGSSolver.minimize function, inside tsm._optimize_fit().
 */
class OptimizerFunction {
private:
    std::function<double(Eigen::VectorXd)> _function;

    double _dfridr(const Eigen::VectorXd& beta, double h, double& err, const int& my_idx);

public:
    explicit OptimizerFunction(std::function<double(Eigen::VectorXd)> function);

    double operator()(const Eigen::VectorXd& beta, Eigen::VectorXd& grad);
};