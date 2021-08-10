#pragma once
#include "../headers.hpp"
#include "family.hpp"

/**
 * @brief Normal distribution for time series
 */
class Normal : Family {
private:
    double mu0;
    double sigma0;
    std::string transform;
    short int param_no;
    bool covariance_prior;
    // gradient_only won't be used (no GAS models)

public:
    /**
     * @brief Constructor for Normal distribution
     * @param mu (double): mean for the Normal distribution
     * @param sigma (double): standard deviation for the Normal distribution
     * @param transform (string): whether to apply a transformation for the location latent variable
     *  (e.g. "exp" or "logit")
     */
    Normal(double mu = 0.0, double sigma = 1.0, std::string transform = "");
};