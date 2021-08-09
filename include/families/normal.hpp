#pragma once
#include "../headers.hpp"

class Normal {
private:
    float mu0;
    float sigma0;
    std::string transform;
    short int param_no;
    bool covariance_prior;
    // gradient_only won't be used (no GAS models)
};