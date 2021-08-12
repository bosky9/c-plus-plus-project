#ifndef C_PLUS_PLUS_PROJECT_FLAT_H
#define C_PLUS_PLUS_PROJECT_FLAT_H
#include "family.hpp"

/**
 * @brief This class contains methods relating to the flat prior distribution for time series
 */
class Flat : Family {
private:
    bool _covariance_prior;

public:
    /**
     * @brief Constructor for Flat
     * @param transform Whether to apply a transformation - e.g. 'exp' or 'logit'
     */
    Flat(const std::string& transform = "");

    /**
     * @brief Copy constructor for Flat distribution
     * @param flat A Flat object
     */
    Flat(const Flat& flat);

    /**
     * @brief Move constructor for Flat distribution
     * @param flat A Flat object
     */
    Flat(Flat&& flat);

    /**
     * @brief Assignment operator for Flat distribution
     * @param flat A Flat object
     */
    Flat& operator=(const Flat& flat);

    /**
     * @brief Move assignment operator for Flat distribution
     * @param flat A Flat object
     */
    Flat& operator=(Flat&& flat);

    /**
     * @brief Log PDF for Flat prior
     * @param mu Latent variable for which the prior is being formed over
     * @return log(p(mu))
     */
    double logpdf(double mu);
};

#endif// C_PLUS_PLUS_PROJECT_FLAT_H