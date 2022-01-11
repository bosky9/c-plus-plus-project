/**
 * @file metropolis_hastings.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core"           // Eigen::VectorXd, Eigen::MatrixXd
#include "inference/sample.hpp" // Sample

#include <optional> // std::optional

/**
 * @class MetropolisHastings metropolis_hastings.hpp
 * @brief Random-walk Metropolis-Hastings MCMC
 */
class MetropolisHastings final {
public:
    /**
     * @brief Constructor for MetropolisHastings
     * @param posterior A posterior function
     * @param scale The scale for the random walk
     * @param nsims The number of iterations to perform
     * @param initials Where to start the MCMC chain
     * @param cov_matrix A covariance matrix for the random walk (optional)
     * @param thinning By how much to thin the chains (2 means drop every other point)
     * @param warm_up_period Whether to discard first half of the chain as 'warm-up'
     * @param quiet_progress Whether to print progress to console or stay quiet
     */
    MetropolisHastings(std::function<double(const Eigen::VectorXd&)>& posterior, double scale, size_t nsims,
                       const Eigen::VectorXd& initials, const std::optional<Eigen::MatrixXd>& cov_matrix = std::nullopt,
                       size_t thinning = 2, bool warm_up_period = true, bool quiet_progress = false);

    /**
     * @brief Tunes scale for M-H algorithm
     * @param acceptance The most recent acceptance rate
     * @param scale The current scale parameter
     * @return An adjusted scale parameter
     */
    static double tune_scale(double acceptance, double scale);

    /**
     * @brief Sample from M-H algorithm
     * @return A Sample object
     */
    Sample sample();

private:
    std::function<double(const Eigen::VectorXd&)> _posterior; ///< A posterior function
    double _scale;                                            ///< The scale for the random walk
    size_t _nsims;                                            ///< The number of iterations to perform
    Eigen::VectorXd _initials;                                ///< Where to start the MCMC chain
    size_t _param_no;                                         ///< Number of parameters
    size_t _thinning;            ///< By how much to thin the chains (2 means drop every other point)
    bool _warm_up_period;        ///< Whether to discard first half of the chain as 'warm-up'
    bool _quiet_progress;        ///< Whether to print progress to console or stay quiet
    Eigen::MatrixXd _phi;        ///< Matrix for the Metropolis-Hastings algorithm
    Eigen::MatrixXd _cov_matrix; ///< A covariance matrix for the random walk
                                 // TSM model not actually used
};