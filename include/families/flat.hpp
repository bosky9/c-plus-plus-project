#pragma once

#include "family.hpp"

/**
 * @brief This class contains methods relating to the flat prior distribution for time series
 */
class Flat final : public Family {
private:
    bool _covariance_prior;

public:
    /**
     * @brief Constructor for Flat
     * @param transform Whether to apply a transformation - e.g. 'exp' or 'logit'
     */
    explicit Flat(const std::string& transform = "");

    /**
     * @brief Copy constructor for Flat distribution
     * @param flat A Flat object
     */
    Flat(const Flat& flat);

    /**
     * @brief Move constructor for Flat distribution
     * @param flat A Flat object
     */
    Flat(Flat&& flat) noexcept;

    /**
     * @brief Assignment operator for Flat distribution
     * @param flat A Flat object
     */
    Flat& operator=(const Flat& flat);

    /**
     * @brief Move assignment operator for Flat distribution
     * @param flat A Flat object
     */
    Flat& operator=(Flat&& flat) noexcept;

    /**
     * @brief Equal operator for Flat
     * @param flat1 First object
     * @param flat2 Second object
     * @return If the two objects are equal
     */
    friend bool operator==(const Flat& flat1, const Flat& flat2);

    /**
     * @brief Log PDF for Flat prior
     * @param mu Latent variable for which the prior is being formed over
     * @return log(p(mu))
     */
    double logpdf(double mu);

    /**
     * @brief Return the covariance prior
     * @return The covariance prior
     */
    [[nodiscard]] bool get_covariance_prior() const;
};
