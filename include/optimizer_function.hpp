#pragma once

#include "Eigen/Core" // Eigen::VectorXd

#include <functional> // std::function

/**
 * @class OptimizerFunction optimizer_function.hpp
 * @brief Used to have a function and a way to compute its derivative in the same class.
 *
 * @details     Necessary for the LBFGSSolver.minimize function, inside tsm._optimize_fit().
 */
class OptimizerFunction final {

public:
    explicit OptimizerFunction(std::function<double(const Eigen::VectorXd&)> function);

    double operator()(const Eigen::VectorXd& beta, Eigen::VectorXd& grad);

private:
    std::function<double(Eigen::VectorXd)> _function;
};