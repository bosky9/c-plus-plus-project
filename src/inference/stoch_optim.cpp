#include "inference/stoch_optim.hpp"

StochOptim::StochOptim(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate)
    : _parameters{starting_parameters}, _variances{starting_variances}, _learning_rate{learning_rate} {}

    Eigen::VectorXd StochOptim::update(Eigen::VectorXd& gradient) {
    return {};
}

Eigen::VectorXd StochOptim::get_parameters() const {
    return _parameters;
}

RMSProp::RMSProp(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
                 double ewma)
    : _ewma{ewma}, StochOptim{starting_parameters, starting_variances, learning_rate} {}

RMSProp::RMSProp(const RMSProp& rmsprop) : StochOptim{rmsprop} {
    _parameters    = rmsprop._parameters;
    _variances      = rmsprop._variances;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
}

RMSProp::RMSProp(RMSProp&& rmsprop) : StochOptim{std::move(rmsprop)} {
    _parameters    = rmsprop._parameters;
    _variances      = rmsprop._variances;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    rmsprop._parameters.resize(0);
    rmsprop._variances.resize(0);
    rmsprop._learning_rate = 0.0;
    rmsprop._ewma          = 0.0;
    rmsprop._t             = 0;
}

RMSProp& RMSProp::operator=(const RMSProp& rmsprop) {
    if (this == &rmsprop)
        return *this;
    _parameters    = rmsprop._parameters;
    _variances      = rmsprop._variances;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    return *this;
}

RMSProp& RMSProp::operator=(RMSProp&& rmsprop) {
    _parameters    = rmsprop._parameters;
    _variances      = rmsprop._variances;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    rmsprop._parameters.resize(0);
    rmsprop._variances.resize(0);
    rmsprop._learning_rate = 0.0;
    rmsprop._ewma          = 0.0;
    rmsprop._t             = 0;
    return *this;
}

Eigen::VectorXd RMSProp::update(Eigen::VectorXd& gradient) {
    _variances = _ewma * _variances + (1 - _ewma) * pow(gradient, 2);
    if (_t > 5) {
        double add_to_param =
                _learning_rate + (_learning_rate * 15.0 * (pow(0.99, _t))) * (gradient / sqrt(_variances + _epsilon));
        std::transform(_parameters.begin(), _parameters.end(), _parameters.begin(),
                       [add_to_param](double val) { return val + add_to_param; });
    }

    _t += 1;
    return _parameters;
}

ADAM::ADAM(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate, double ewma1,
           double ewma2)
    : _ewma_1{ewma1}, _ewma_2{ewma2}, StochOptim{starting_parameters, starting_variances, learning_rate} {}

ADAM::ADAM(const ADAM& adam) : StochOptim{adam} {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variances      = adam._variances;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
}

ADAM::ADAM(ADAM&& adam) : StochOptim{std::move(adam)} {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variances      = adam._variances;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    adam._parameters.resize(0);
    adam._f_gradient    = 0.0;
    adam._variances.resize(0);
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
    _variances      = adam._variances;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    return *this;
}

ADAM& ADAM::operator=(ADAM&& adam) {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variances      = adam._variances;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    adam._parameters.resize(0);
    adam._f_gradient    = 0.0;
    adam._variances.resize(0);
    adam._learning_rate = 0.0;
    adam._ewma_1        = 0.0;
    adam._ewma_2        = 0.0;
    adam._t             = 0;
    return *this;
}

Eigen::VectorXd ADAM::update(Eigen::VectorXd& gradient) {
    _f_gradient           = _ewma_1 * _f_gradient + (1 - _ewma_1) * gradient;
    double f_gradient_hat = _f_gradient / (1 - pow(_ewma_1, _t));
    _variances             = _ewma_2 * _variances + (1 - _ewma_2) * pow(gradient, 2);
    Eigen::VectorXd variance_hats   = _variances / (1 - pow(_ewma_2, _t));
    if (_t > 5) {
        double add_to_param = _learning_rate + (_learning_rate * 15.0 * (pow(0.99, _t))) *
                                                       (f_gradient_hat / (sqrt(variance_hats) + _epsilon));
        std::transform(_parameters.begin(), _parameters.end(), _parameters.begin(),
                       [add_to_param](double val) { return val + add_to_param; });
    }
    _t += 1;
    return _parameters;
}