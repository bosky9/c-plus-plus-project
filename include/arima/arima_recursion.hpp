#pragma once

#include "headers.hpp"

/**
 * @brief Max function between two doubles
 * @param a First value
 * @param b Second value
 * @return Maximum number between a and b
 */
inline double double_max(double a, double b) {
    return a ? a >= b : b;
}

/**
 * @brief Min function between two doubles
 * @param a First value
 * @param b Second value
 * @return Minimum number between a and b
 */
inline double double_min(double a, double b) {
    return a ? a <= b : b;
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
Eigen::VectorXd arima_recursion(Eigen::VectorXd parameters, Eigen::VectorXd mu, Eigen::VectorXd link_mu,
                                Eigen::VectorXd Y, size_t max_lag, size_t Y_len, size_t ar_terms, size_t ma_terms);

// TODO: Non abbiamo implementato in families Poisson!
Eigen::VectorXd arima_recursion_poisson(Eigen::VectorXd parameters, Eigen::VectorXd mu, Eigen::VectorXd link_mu,
                                        Eigen::VectorXd Y, size_t max_lag, size_t Y_len, size_t ar_terms,
                                        size_t ma_terms);

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
Eigen::VectorXd arima_recursion_normal(Eigen::VectorXd parameters, Eigen::VectorXd mu, Eigen::VectorXd Y,
                                       size_t max_lag, size_t Y_len, size_t ar_terms, size_t ma_terms);

// TODO: Da implementare solo in caso si faccia la classe ARIMAX
Eigen::VectorXd arimax_recursion(Eigen::VectorXd parameters, Eigen::VectorXd mu, Eigen::VectorXd Y, size_t max_lag,
                                 size_t Y_len, size_t ar_terms, size_t ma_terms);