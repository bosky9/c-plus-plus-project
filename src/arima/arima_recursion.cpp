#include "arima/arima_recursion.hpp"

Eigen::VectorXd arima_recursion(Eigen::VectorXd parameters, Eigen::VectorXd mu, Eigen::VectorXd link_mu,
                                Eigen::VectorXd Y, size_t max_lag, size_t Y_len, size_t ar_terms, size_t ma_terms) {
    for (auto i{static_cast<Eigen::Index>(max_lag)}; i < Y_len; i++) {
        for (Eigen::Index j{0}; j < ma_terms; j++)
            mu[i] += parameters[1 + static_cast<Eigen::Index>(ar_terms) + j] * (Y[i - 1 - j] - link_mu[i - 1 - j]);
    }
    return mu;
}

Eigen::VectorXd arima_recursion_normal(Eigen::VectorXd parameters, Eigen::VectorXd mu, Eigen::VectorXd Y,
                                       size_t max_lag, size_t Y_len, size_t ar_terms, size_t ma_terms) {
    for (auto i{static_cast<Eigen::Index>(max_lag)}; i < static_cast<Eigen::Index>(Y_len); i++) {
        for (Eigen::Index j{0}; j < ma_terms; j++)
            mu[i] += parameters[1 + static_cast<Eigen::Index>(ar_terms) + j] * (Y[i - 1 - j] - mu[i - 1 - j]);
    }
    return mu;
}