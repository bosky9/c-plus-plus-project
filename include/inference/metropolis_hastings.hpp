#pragma once

#include <Eigen/Core>

#include "sample.hpp"

/**
 * @brief RANDOM-WALK METROPOLIS-HASTINGS MCMC
 */
class MetropolisHastings {
private:
    /// A posterior function
    std::function<double(Eigen::VectorXd)> posterior;

    /// The scale for the random walk
    double scale;

    /// The number of iterations to perform
    int nsims;

    /// Where to start the MCMC chain
    Eigen::VectorXd initials;

    /// (optional) A covariance matrix for the random walk
    Eigen::MatrixXd cov_matrix;

    /// By how much to thin the chains (2 means drop every other point)
    int thinning;

    /// Whether to discard first half of the chain as 'warm-up'
    bool warm_up_period;

    /// A model object (for use in SPDK sampling)
    // TODO: TSM model_object;

    /// Whether to print progress to console or stay quiet
    bool quiet_progress;
public:
    MetropolisHastings(std::function<double(Eigen::VectorXd)> posterior,
                       double scale,
                       int nsims,
                       Eigen::VectorXd& initials,
                       // TODO: La reference non può essere nullptr
                       Eigen::MatrixXd& cov_matrix = nullptr,
                       int thinning = 2,
                       bool warm_up_period = true,
                       // TODO: TSM model_object = nullptr,
                       bool quiet_progress = false
    );

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
};