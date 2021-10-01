#pragma once

#include "headers.hpp"

const double h = 0.00001;

[[nodiscard]] Eigen::MatrixXd hessian(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v);

[[nodiscard]] double first_derivative(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v,
                                      Eigen::Index i);

[[nodiscard]] double second_derivative(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v,
                                       Eigen::Index i, Eigen::Index j);