/**
 * @file stoch_optim.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/stoch_optim.hpp"

double StochOptim::_epsilon = pow(10.0, -8);

StochOptim::StochOptim(Eigen::VectorXd starting_parameters, Eigen::VectorXd starting_variances, double learning_rate)
    : _parameters{std::move(starting_parameters)}, _variances{std::move(starting_variances)}, _learning_rate{
                                                                                                      learning_rate} {}

Eigen::VectorXd StochOptim::update([[maybe_unused]] Eigen::VectorXd& gradient) {
    return {};
}

Eigen::VectorXd StochOptim::get_parameters() const {
    return _parameters;
}

RMSProp::RMSProp(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances,
                 double learning_rate, double ewma)
    : StochOptim{starting_parameters, starting_variances, learning_rate}, _ewma{ewma} {}

Eigen::VectorXd RMSProp::update(Eigen::VectorXd& gradient) {
    _variances = _ewma * _variances.array() + (1 - _ewma) * pow(gradient.array(), 2);
    if (_t > 5) {
        _parameters =
                _parameters.array() + (_learning_rate +
                (_learning_rate * 15.0 * (pow(0.99, _t)))) * (gradient.array() / sqrt(_variances.array() + _epsilon));
    }
    _t += 1;
    return _parameters;
}

ADAM::ADAM(const Eigen::VectorXd& starting_parameters, const Eigen::VectorXd& starting_variances, double learning_rate,
           double ewma1, double ewma2)
    : StochOptim{starting_parameters, starting_variances, learning_rate},
      _f_gradient{Eigen::VectorXd::Zero(_parameters.size())}, _ewma_1{ewma1}, _ewma_2{ewma2} {}

Eigen::VectorXd ADAM::update(Eigen::VectorXd& gradient) {
    _f_gradient                    = _ewma_1 * _f_gradient.array() + (1 - _ewma_1) * gradient.array();
    Eigen::VectorXd f_gradient_hat = _f_gradient / (1 - pow(_ewma_1, _t));
    _variances                     = _ewma_2 * _variances.array() + (1 - _ewma_2) * pow(gradient.array(), 2);
    Eigen::VectorXd variance_hats  = _variances / (1 - pow(_ewma_2, _t));
    if (_t > 5) {
        _parameters = _parameters.array() + (_learning_rate +
                      (_learning_rate * 15.0 * (pow(0.99, _t)))) *
                              (f_gradient_hat.array() / (sqrt(variance_hats.array()) + _epsilon));
    }
    _t += 1;
    return _parameters;
}