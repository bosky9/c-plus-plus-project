#include "families/normal.hpp"

Normal::Normal(double mu, double sigma, std::string transform) : Family(transform), mu0{mu}, sigma0{sigma},
    param_no{2}, covariance_prior{false} {}
