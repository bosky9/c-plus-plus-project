#pragma once

#include <algorithm>

#include "headers.hpp"
#include "multivariate_normal.hpp"
#include "sample.hpp"

/**
 * @brief Number of simulations
 */
const int NSIMS = 30000;

/**
 * @brief 95-th percentile of simulations
 */
const int N95 = NSIMS * 95 / 100;

/**
 * @brief 5-th percentile of simulations
 */
const int N5 = NSIMS * 5 / 100;

// 1 dispari    (true)
// 0 pari       (false)
/**
 * @brief Compile time function that returns if NSIMS is odd
 */
constexpr bool NSIMS_ODD() {
    return static_cast<bool>(NSIMS % 2);
}

/**
 * @brief
 * @param modes Modes vector
 * @param cov_matrix Covariances matrix
 * @return A NormPostSimData structure
 */
inline Sample norm_post_sim(const Eigen::VectorXd& modes, const Eigen::MatrixXd& cov_matrix) {
    Mvn mul_norm{modes, cov_matrix};
    Eigen::Index modes_len = modes.size();
    Eigen::Matrix<double, NSIMS, Eigen::Dynamic> phi =
            Eigen::Matrix<double, NSIMS, Eigen::Dynamic>::Zero(NSIMS, modes_len);

    for (Eigen::Index i{0}; i < NSIMS; i++) {
        phi.row(i) = mul_norm.sample(); // Incollo un vettore di lunghezza modes.size() in ogni riga
    }

    Eigen::MatrixXd chain = phi.transpose();

    Eigen::VectorXd mean_vector = phi.colwise().mean();
    Eigen::VectorXd mean_est(mean_vector.size());
    Eigen::VectorXd::Map(&mean_est[0], mean_vector.size()) = mean_vector;

    std::vector<double> median_est{}, upper_95_est{}, lower_5_est{};
    for (int i = 0; i < modes_len; i++) {
        std::vector<double> col_sort(NSIMS);
        Eigen::VectorXd::Map(&col_sort[0], NSIMS) = phi.col(i);
        std::sort(col_sort.begin(), col_sort.end());

        if (NSIMS_ODD)
            median_est.push_back(col_sort[NSIMS / 2]);
        else
            median_est.push_back((col_sort[NSIMS / 2] + col_sort[NSIMS / 2 - 1]) / 2);

        upper_95_est.push_back(col_sort[N95]);
        lower_5_est.push_back(col_sort[N5]);
    }

    return {chain, mean_est, Eigen::VectorXd::Map(median_est.data(), median_est.size()),
            Eigen::VectorXd::Map(upper_95_est.data(), upper_95_est.size()),
            Eigen::VectorXd::Map(lower_5_est.data(), lower_5_est.size())};
}