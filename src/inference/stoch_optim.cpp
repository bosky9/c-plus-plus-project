#include "inference/stoch_optim.hpp"

RMSProp::RMSProp(const std::vector<double>& starting_parameters, double starting_variance, double learning_rate,
        double ewma) :
        _parameters{starting_parameters},
        _variance{starting_variance},
        _learning_rate{learning_rate},
        _ewma{ewma} {};

std::vector<double> RMSProp::update(double gradient) {
    _variance = _ewma * _variance + (1 - _ewma) * pow(gradient, 2);
    
};


ADAM::ADAM(const std::vector<double>& starting_parameters, double starting_variance, double learning_rate,
     double ewma1, double ewma2) :
     _parameters{starting_parameters},
     _variance{starting_variance},
     _learning_rate{learning_rate},
     _ewma_1{ewma1},
     _ewma_2{ewma2} {};

std::vector<double> ADAM::update(double gradient) {
    _f_gradient = _ewma_1 * _f_gradient + (1 - _ewma_1) * gradient;
    double f_gradient_hat = _f_gradient / (1 - pow(_ewma_1, _t));
    _variance = _ewma_2 * _variance + (1 - _ewma_2) * pow(gradient, 2);
    double variance_hat = _variance / (1 - pow(_ewma_2, _t));
    if (_t > 5) {
        double add_to_param = _learning_rate + (_learning_rate*15.0*(pow(0.99, _t))) * (f_gradient_hat/(sqrt(variance_hat)+_epsilon));
        std::transform(_parameters.begin(), _parameters.end(), _parameters.begin(),
                       [add_to_param](double val){return val + add_to_param});
    }
    _t += 1;
    return _parameters;
};