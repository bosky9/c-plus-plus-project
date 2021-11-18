/**
 * @file bbvi_routines.hpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd

#include <optional> // std::optional

/**
 * @brief Namespace for BBVI routines
 */
namespace bbvi_routines {

/**
 * @brief Compute covariance of two vectors
 * @param x First vector of doubles
 * @param y Second vector of doubles
 * @return Covariance
 */
double covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

/**
 * @brief Alpha recursion for BBVI class
 * @param alpha0 Set of admissible ordinals
 * @param grad_log_q Gradients
 * @param gradient Gradients
 * @param param_no Number of parameters
 */
void alpha_recursion(Eigen::VectorXd& alpha0, const Eigen::MatrixXd& grad_log_q, const Eigen::MatrixXd& gradient,
                     uint8_t param_no);

/**
 * @brief Posterior for BBVI class
 * @param z A matrix from which to extract the vectors (rows) used in neg_posterior
 * @param neg_posterior A function which takes a vector (Eigen::VectorXd) and returns a double
 * @return A vector of doubles
 */
Eigen::VectorXd
log_p_posterior(const Eigen::MatrixXd& z,
                const std::function<double(const Eigen::VectorXd&, std::optional<size_t>)>& neg_posterior);

/**
 * @brief Posterior for BBVI class with mini batch
 * @param z A matrix from which to extract the vectors (rows) used in neg_posterior
 * @param neg_posterior A function which takes a vector (Eigen::VectorXd) and an integer, and returns a double
 * @param mini_batch Number of mini batches
 * @return An array of doubles
 */
Eigen::VectorXd
mb_log_p_posterior(const Eigen::MatrixXd& z,
                   const std::function<double(const Eigen::VectorXd&, std::optional<size_t>)>& neg_posterior,
                   size_t mini_batch);

} // namespace bbvi_routines