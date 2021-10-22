#pragma once

#include "arima/arima.hpp"

#include <Eigen/Core>
#include <functional>
#include <iostream>

class OptimizerFunction {
private:
    std::function<double(Eigen::VectorXd)> _function;

public:
    explicit OptimizerFunction(std::function<double(Eigen::VectorXd)> function);

    double operator()(const Eigen::VectorXd& beta, Eigen::VectorXd& grad);
};