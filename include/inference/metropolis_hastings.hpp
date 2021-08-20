#pragma once

#include <Eigen/Core>

#include "sample.hpp"

/**
 * @brief RANDOM-WALK METROPOLIS-HASTINGS MCMC
 */
class MetropolisHastings {
private:
    std::function<double(Eigen::VectorXd)> posterior;
    double scale;
    int nsims;
    Eigen::VectorXd initials;
    Eigen::MatrixXd cov_matrix;
    int thinning;
    bool warm_up_period;
    // TODO: TSM model_object;
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

    static double tune_scale(double acceptance, double scale);

    Sample sample();
};