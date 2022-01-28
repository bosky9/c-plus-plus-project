/**
 * @file arima_recursion.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd

/**
 * @brief Max function between two doubles
 * @param a First value
 * @param b Second value
 * @return Maximum number between a and b
 */
[[maybe_unused]] inline double double_max(double a, double b) {
    return (a >= b) ? a : b;
}

/**
 * @brief Min function between two doubles
 * @param a First value
 * @param b Second value
 * @return Minimum number between a and b
 */
[[maybe_unused]] inline double double_min(double a, double b) {
    return (a <= b) ? a : b;
}

/**
 * @brief Moving average recursion for ARIMA model class
 * @param parameters
 * @param mu
 * @param link_mu
 * @param Y
 * @param max_lag
 * @param Y_len
 * @param ar_terms
 * @param ma_terms
 * @return
 */
void arima_recursion(const Eigen::VectorXd& parameters, Eigen::VectorXd& mu, const Eigen::VectorXd& link_mu,
                     const Eigen::VectorXd& Y, size_t max_lag, size_t Y_len, size_t ar_terms, size_t ma_terms);

/**
 * @brief Moving average recursion for ARIMA model class - Gaussian errors
 * @param parameters
 * @param mu
 * @param link_mu
 * @param Y
 * @param max_lag
 * @param Y_len
 * @param ar_terms
 * @param ma_terms
 * @return
 */
void arima_recursion_normal(const Eigen::VectorXd& parameters, Eigen::VectorXd& mu, const Eigen::VectorXd& Y,
                            size_t max_lag, size_t Y_len, size_t ar_terms, size_t ma_terms);