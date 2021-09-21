#pragma once

#include "multivariate_normal.hpp"

inline double find_p_value(const double z) {
    double p_value = 0.0;
    if (z >= 0.0) {
        p_value += 1 - Mvn::cdf(z);
        p_value += Mvn::cdf(-z);
    } else {
        p_value += 1 - Mvn::cdf(-z);
        p_value += Mvn::cdf(z);
    }
    return p_value;
}