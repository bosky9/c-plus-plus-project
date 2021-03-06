/**
 * @file flat.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "families/family.hpp" // Family

#include <memory>              // std::unique_ptr
#include <string>              // std::string

/**
 * @class Flat flat.hpp
 * @brief This class contains methods relating to the flat prior distribution for time series
 */
class Flat final : public Family {
public:
    /**
     * @brief Constructor for Flat
     * @param transform Whether to apply a transformation - e.g. 'exp' or '_logit'
     */
    explicit Flat(const std::string& transform = "");

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
    [[nodiscard]] double logpdf(double mu) const override;

    // Get methods -----------------------------------------------------------------------------------------------------

    /**
     * @brief Get the name of the distribution family for the get_z_priors_names() method of LatentVariables
     * @return Name of the distribution family
     */
    [[nodiscard]] std::string get_name() const override;

    /**
     * @brief Get the description of the parameters of the distribution family for the get_z_priors_names() method of
     * LatentVariables
     * @return Description of the parameters of the distribution family
     */
    [[nodiscard]] std::string get_z_name() const override;

    // Clone function --------------------------------------------------------------------------------------------------

    /**
     * @brief Returns a clone of the current object
     * @return A copy of the family object which calls this function
     *
     * @details Overrides the family one, returns a new Flat object by deep copy of the current one.
     */
    [[nodiscard]] std::unique_ptr<Family> clone() const override;

private:
    bool _covariance_prior; ///< Covariance's prior
};
