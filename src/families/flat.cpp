#include "../../include/families/flat.hpp"

Flat::Flat(std::string transform) : Family(transform), covariance_prior(false) {}

double Flat::logpdf(double mu) {
    return 0.0;
}