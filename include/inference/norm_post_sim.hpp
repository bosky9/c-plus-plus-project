/**
 * @file norm_post_sim.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include "sample.hpp" // Sample

namespace nps {

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

/**
 * @brief Compile time function that returns if NSIMS is odd
 * @return True if NSIMS is odd, false otherwise
 */
constexpr bool NSIMS_ODD() {
    return static_cast<bool>(NSIMS % 2);
}

/**
 * @brief Function useful for Results' constructor
 * @param modes Modes vector
 * @param cov_matrix Covariances matrix
 * @return A Sample object
 */
Sample norm_post_sim(const Eigen::VectorXd& modes, const Eigen::MatrixXd& cov_matrix);

} // namespace nps