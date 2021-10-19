#pragma once

#include "headers.hpp"
#include "inference/metropolis_sampler.hpp"
#include "multivariate_normal.hpp"
#include "sample.hpp"

#include <optional>

/**
 * @brief Random-walk Metropolis-Hastings MCMC
 */
class MetropolisHastings final {
private:
    std::function<double(Eigen::VectorXd)> _posterior; ///< A posterior function
    double _scale;                                     ///< The scale for the random walk
    size_t _nsims;                                     ///< The number of iterations to perform
    Eigen::VectorXd _initials;                         ///< Where to start the MCMC chain
    Eigen::Index _param_no;                            ///< Number of paramters
    int _thinning;               ///< By how much to thin the chains (2 means drop every other point)
    bool _warm_up_period;        ///< Whether to discard first half of the chain as 'warm-up'
    bool _quiet_progress;        ///< Whether to print progress to console or stay quiet
    Eigen::MatrixXd _phi;        ///< Matrix of...
    Eigen::MatrixXd _cov_matrix; ///< A covariance matrix for the random walk
    // TODO: TSM model; ///< A model object (for use in SPDK sampling)

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
     * // TODO: TSM model_object A model object (for use in SPDK sampling)
     * @param quiet_progress Whether to print progress to console or stay quiet
     */
    MetropolisHastings(std::function<double(const Eigen::VectorXd&)>& posterior, double scale, size_t nsims,
                       const Eigen::VectorXd& initials, const std::optional<Eigen::MatrixXd>& cov_matrix = std::nullopt,
                       int thinning = 2, bool warm_up_period = true, // TODO: TSM model_object = nullptr,
                       bool quiet_progress = false);

    MetropolisHastings(const MetropolisHastings& mh);

    /**
     * @brief Move constructor for MetropolisHastings
     * @param mh A MetropolisHastings object
     */
    MetropolisHastings(MetropolisHastings&& mh);

    /**
     * @brief Assignment operator for MetropolisHastings
     * @param mh A MetropolisHastings object
     */
    MetropolisHastings& operator=(const MetropolisHastings& mh);

    /**
     * @brief Move assignment operator for MetropolisHastings
     * @param mh A MetropolisHastings object
     */
    MetropolisHastings& operator=(MetropolisHastings&& mh);

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