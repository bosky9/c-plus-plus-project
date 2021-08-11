#ifndef C_PLUS_PLUS_PROJECT_FLAT_H
#define C_PLUS_PLUS_PROJECT_FLAT_H
#include "family.hpp"

/**
 * @brief This class contains methods relating to the flat prior distribution for time series.
 */
class Flat : Family {
private:
    bool covariance_prior;
public:
    Flat(std::string transform = "");
    double logpdf(double mu);
};

#endif //C_PLUS_PLUS_PROJECT_FLAT_H