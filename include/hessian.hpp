/**
 * @file hessian.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd

#include <functional> // std::function

namespace derivatives {

const double h = 0.00001;

/** @brief Approximation of pthon numdifftools.Hessian
 * @param function The function do derive
 * @param v The point where to compute the derivative.
 *
 * @details The approximation error is, according to wikipedia,
 *          -(f^3(c)) * h^2 / 6, with c being some point between x+h, x-h,
 *          x being the point where the derivative is computed.
 */
[[nodiscard]] Eigen::MatrixXd hessian(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v);

/**
 * @brief First derivative
 * @param function The function do derive
 * @param v The point where to compute the derivative
 * @param i
 * @return First derivative
 */
[[nodiscard]] double first_derivative(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v,
                                      Eigen::Index i);

/**
 * @brief Second derivative
 * @param function The function do derive
 * @param v The point where to compute the derivative
 * @param i
 * @param j
 * @return Second derivative
 */
[[nodiscard]] double second_derivative(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v,
                                       Eigen::Index i, Eigen::Index j);

} // namespace derivatives