#include "inference/stoch_optim.hpp"

RMSProp::RMSProp(const std::vector<double>& starting_parameters, double starting_variance, double learning_rate,
                 double ewma)
    : _parameters{starting_parameters}, _variance{starting_variance}, _learning_rate{learning_rate}, _ewma{ewma} {};

RMSProp::RMSProp(const RMSProp& rmsprop) {
    _parameters    = rmsprop._parameters;
    _variance      = rmsprop._variance;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
}

RMSProp::RMSProp(RMSProp&& rmsprop) {
    _parameters    = rmsprop._parameters;
    _variance      = rmsprop._variance;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    rmsprop._parameters.clear();
    rmsprop._variance      = 0.0;
    rmsprop._learning_rate = 0.0;
    rmsprop._ewma          = 0.0;
    rmsprop._t             = 0;
}

RMSProp& RMSProp::operator=(const RMSProp& rmsprop) {
    if (this == &rmsprop)
        return *this;
    _parameters    = rmsprop._parameters;
    _variance      = rmsprop._variance;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    return *this;
}

RMSProp& RMSProp::operator=(RMSProp&& rmsprop) {
    _parameters    = rmsprop._parameters;
    _variance      = rmsprop._variance;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    rmsprop._parameters.clear();
    rmsprop._variance      = 0.0;
    rmsprop._learning_rate = 0.0;
    rmsprop._ewma          = 0.0;
    rmsprop._t             = 0;
    return *this;
}

std::vector<double> RMSProp::update(double gradient) {
    _variance = _ewma * _variance + (1 - _ewma) * pow(gradient, 2);
    if (_t > 5) {
        double add_to_param =
                _learning_rate + (_learning_rate * 15.0 * (pow(0.99, _t))) * (gradient / sqrt(_variance + _epsilon));
        std::transform(_parameters.begin(), _parameters.end(), _parameters.begin(),
                       [add_to_param](double val) { return val + add_to_param; });
    }

    _t += 1;
    return _parameters;
}

ADAM::ADAM(const std::vector<double>& starting_parameters, double starting_variance, double learning_rate, double ewma1,
           double ewma2)
    : _parameters{starting_parameters}, _variance{starting_variance},
      _learning_rate{learning_rate}, _ewma_1{ewma1}, _ewma_2{ewma2} {};

ADAM::ADAM(const ADAM& adam) {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variance      = adam._variance;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
}

ADAM::ADAM(ADAM&& adam) {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variance      = adam._variance;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    adam._parameters.clear();
    adam._f_gradient    = 0.0;
    adam._variance      = 0.0;
    adam._learning_rate = 0.0;
    adam._ewma_1        = 0.0;
    adam._ewma_2        = 0.0;
    adam._t             = 0;
}

ADAM& ADAM::operator=(const ADAM& adam) {
    if (this == &adam)
        return *this;
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variance      = adam._variance;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    return *this;
}

ADAM& ADAM::operator=(ADAM&& adam) {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variance      = adam._variance;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    adam._parameters.clear();
    adam._f_gradient    = 0.0;
    adam._variance      = 0.0;
    adam._learning_rate = 0.0;
    adam._ewma_1        = 0.0;
    adam._ewma_2        = 0.0;
    adam._t             = 0;
    return *this;
}

std::vector<double> ADAM::update(double gradient) {
    _f_gradient           = _ewma_1 * _f_gradient + (1 - _ewma_1) * gradient;
    double f_gradient_hat = _f_gradient / (1 - pow(_ewma_1, _t));
    _variance             = _ewma_2 * _variance + (1 - _ewma_2) * pow(gradient, 2);
    double variance_hat   = _variance / (1 - pow(_ewma_2, _t));
    if (_t > 5) {
        double add_to_param = _learning_rate + (_learning_rate * 15.0 * (pow(0.99, _t))) *
                                                       (f_gradient_hat / (sqrt(variance_hat) + _epsilon));
        std::transform(_parameters.begin(), _parameters.end(), _parameters.begin(),
                       [add_to_param](double val) { return val + add_to_param; });
    }
    _t += 1;
    return _parameters;
};