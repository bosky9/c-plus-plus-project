#include "optimizer_function.hpp"

#include <iostream>
#include <utility>

OptimizerFunction::OptimizerFunction(std::function<double(Eigen::VectorXd)>  function) : _function{std::move(function)} {}

double OptimizerFunction::operator()(const Eigen::VectorXd& beta, Eigen::VectorXd& grad) {
    // Copies of beta
    Eigen::VectorXd beta_temp_plus = beta;
    Eigen::VectorXd beta_temp_minus = beta;

    for (int i = 0; i < beta.size(); i++){
        // Init h
        double h;
        if(beta[i] != 0) {
            h             = std::cbrt(std::numeric_limits<double>::epsilon()) * beta[i];
            double volatile temp = beta[i] + h;
            h                    = temp - beta[i];
        }
        else{
            h             = std::cbrt(std::numeric_limits<double>::epsilon()) * 1e-6;
            double volatile temp = 1e-6 + h;
            h                    = temp - 1e-6;
        }

        // Add h
        beta_temp_plus[i]   = beta[i] + h;
        beta_temp_minus[i]  = beta[i] - h;

        // Create f(x+h), f(x-h)
        double f_plus = _function(beta_temp_plus);
        double f_minus = _function(beta_temp_minus);

        // Compute gradient
        grad[i] = (f_plus - f_minus)/(2*h);

        // Refresh copy vectors
        beta_temp_plus[i]   = beta[i];
        beta_temp_minus[i]  = beta[i];
    }

    // Compute actual value
    return _function(beta);
}