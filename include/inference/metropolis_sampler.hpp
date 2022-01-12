/**
 * @file metropolis_sampler.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd

/**
 * @brief Metropolis namespace including (only) metropolis_sampler function
 */
namespace metropolis {

/**
 * @brief Sample a number of simulations for Metropolis Hastings
 * @param sims_to_do Number of iterations
 * @param phi Matrix to fill
 * @param posterior Posterior function
 * @param a_rate Acceptance rates
 * @param rnums Samples of multivariate uniform distributions
 * @param crit Random numbers
 */
void metropolis_sampler(size_t sims_to_do, Eigen::MatrixXd& phi,
                        const std::function<double(const Eigen::VectorXd&)>& posterior, Eigen::VectorXd& a_rate,
                        const Eigen::MatrixXd& rnums, const Eigen::VectorXd& crit);

} // namespace metropolis