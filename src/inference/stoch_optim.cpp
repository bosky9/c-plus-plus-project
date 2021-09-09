#include "inference/stoch_optim.hpp"

StochOptim::StochOptim(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances,
                       double learning_rate)
    : _parameters{starting_parameters}, _variances{starting_variances}, _learning_rate{learning_rate} {}

StochOptim::StochOptim(const StochOptim& stochOptim) = default;

StochOptim::StochOptim(StochOptim&& stochOptim) noexcept : StochOptim(stochOptim) {
    stochOptim._parameters.resize(0);
    stochOptim._variances.resize(0);
    stochOptim._learning_rate = 0.0;
    stochOptim._t             = 0;
}

StochOptim& StochOptim::operator=(const StochOptim& stochOptim) {
    if (this == &stochOptim)
        return *this;
    _parameters    = stochOptim._parameters;
    _variances     = stochOptim._variances;
    _learning_rate = stochOptim._learning_rate;
    _t             = stochOptim._t;
    return *this;
}

StochOptim& StochOptim::operator=(StochOptim&& stochOptim) noexcept {
    _parameters    = stochOptim._parameters;
    _variances     = stochOptim._variances;
    _learning_rate = stochOptim._learning_rate;
    _t             = stochOptim._t;
    stochOptim._parameters.resize(0);
    stochOptim._variances.resize(0);
    stochOptim._learning_rate = 0.0;
    stochOptim._t             = 0;
    return *this;
}

Eigen::VectorXd StochOptim::update(Eigen::VectorXd& gradient) {
    return {};
}

Eigen::VectorXd StochOptim::get_parameters() const {
    return _parameters;
}

RMSProp::RMSProp(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances,
                 double learning_rate, double ewma)
    : _ewma{ewma}, StochOptim{starting_parameters, starting_variances, learning_rate} {}

RMSProp::RMSProp(const RMSProp& rmsprop) = default;

RMSProp::RMSProp(RMSProp&& rmsprop) noexcept : StochOptim{std::move(rmsprop)}, _ewma{rmsprop._ewma} {
    rmsprop._ewma = 0.0;
}

RMSProp& RMSProp::operator=(const RMSProp& rmsprop) {
    if (this == &rmsprop)
        return *this;
    _parameters    = rmsprop._parameters;
    _variances     = rmsprop._variances;
    _learning_rate = rmsprop._learning_rate;
    _ewma          = rmsprop._ewma;
    _t             = rmsprop._t;
    return *this;
}

RMSProp& RMSProp::operator=(RMSProp&& rmsprop) noexcept {
    _parameters    = rmsprop._parameters;
    _variances     = rmsprop._variances;
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
    _variances = _ewma * _variances.array() + (1 - _ewma) * pow(gradient.array(), 2);
    if (_t > 5) {
        _parameters =
                _parameters.array() + _learning_rate +
                (_learning_rate * 15.0 * (pow(0.99, _t))) * (gradient.array() / sqrt(_variances.array() + _epsilon));
    }

    _t += 1;
    return _parameters;
}

ADAM::ADAM(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
           double ewma1, double ewma2)
    : StochOptim{starting_parameters, starting_variances, learning_rate},
      _f_gradient{Eigen::VectorXd::Zero(_parameters.size())}, _ewma_1{ewma1}, _ewma_2{ewma2} {}

ADAM::ADAM(const ADAM& adam) = default;

ADAM::ADAM(ADAM&& adam) noexcept
    : StochOptim{std::move(adam)}, _f_gradient{adam._f_gradient}, _ewma_1{adam._ewma_1}, _ewma_2{adam._ewma_2} {
    adam._f_gradient = Eigen::VectorXd::Zero(0);
    adam._ewma_1     = 0.0;
    adam._ewma_2     = 0.0;
}

ADAM& ADAM::operator=(const ADAM& adam) {
    if (this == &adam)
        return *this;
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variances     = adam._variances;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    return *this;
}

ADAM& ADAM::operator=(ADAM&& adam) noexcept {
    _parameters    = adam._parameters;
    _f_gradient    = adam._f_gradient;
    _variances     = adam._variances;
    _learning_rate = adam._learning_rate;
    _ewma_1        = adam._ewma_1;
    _ewma_2        = adam._ewma_2;
    _t             = adam._t;
    adam._parameters.resize(0);
    adam._f_gradient = Eigen::VectorXd::Zero(0);
    adam._variances.resize(0);
    adam._learning_rate = 0.0;
    adam._ewma_1        = 0.0;
    adam._ewma_2        = 0.0;
    adam._t             = 0;
    return *this;
}

Eigen::VectorXd ADAM::update(Eigen::VectorXd& gradient) {
    _f_gradient                    = _ewma_1 * _f_gradient.array() + (1 - _ewma_1) * gradient.array();
    Eigen::VectorXd f_gradient_hat = _f_gradient / (1 - pow(_ewma_1, _t));
    _variances                     = _ewma_2 * _variances.array() + (1 - _ewma_2) * pow(gradient.array(), 2);
    Eigen::VectorXd variance_hats  = _variances / (1 - pow(_ewma_2, _t));
    if (_t > 5) {
        _parameters = _parameters.array() + _learning_rate +
                      (_learning_rate * 15.0 * (pow(0.99, _t))) *
                              (f_gradient_hat.array() / (sqrt(variance_hats.array()) + _epsilon));
    }
    _t += 1;
    return _parameters;
}