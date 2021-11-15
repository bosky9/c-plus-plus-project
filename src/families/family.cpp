/**
 * @file family.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "families/family.hpp"

const std::string Family::TRANSFORM_EXP    = "exp";
const std::string Family::TRANSFORM_TANH   = "tanh";
const std::string Family::TRANSFORM_LOGIT  = "logit";
const std::string Family::ITRANSFORM_EXP   = "log";
const std::string Family::ITRANSFORM_TANH  = "arctanh";
const std::string Family::ITRANSFORM_LOGIT = "ilogit";

Family::Family(const std::string& transform) : _transform_name{transform} {
    assert((transform.empty() || transform == TRANSFORM_EXP || transform == TRANSFORM_TANH ||
            transform == TRANSFORM_LOGIT) &&
           "There is no Family with that name.");
    _transform       = _transform_define(transform);
    _itransform_name = _itransform_name_define(transform);
    _itransform      = _itransform_define(transform);
}

bool operator==(const Family& family1, const Family& family2) {
    return family1._transform_name == family2._transform_name && family1._itransform_name == family2._itransform_name;
}

// Get methods -----------------------------------------------------------------------------------------------------

std::string Family::get_name() const {
    return "Prior distribution not detected";
}

std::function<double(double)> Family::get_itransform() const {
    return _itransform;
}

std::string Family::get_itransform_name() const {
    return _itransform_name;
}

std::function<double(double)> Family::get_transform() const {
    return _transform;
}

std::string Family::get_transform_name() const {
    return _transform_name;
}

// Virtual functions for subclasses --------------------------------------------------------------------------------

std::vector<lv_to_build> Family::build_latent_variables() const {
    return {};
}

std::unique_ptr<Family> Family::clone() const {
    return std::make_unique<Family>(*this);
}

Eigen::VectorXd Family::draw_variable(double loc, double scale, int64_t nsims) const {
    return {};
}

Eigen::VectorXd Family::draw_variable(const Eigen::VectorXd& loc, double scale, int64_t nsims) const {
    return {};
}

Eigen::VectorXd Family::draw_variable_local(int64_t size) const {
    return {};
}

uint8_t Family::get_param_no() const {
    return 0;
}

std::string Family::get_z_name() const {
    return "";
}

double Family::logpdf(double mu) const {
    return {};
}

double Family::neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale) const {
    return 0;
}

FamilyAttributes Family::setup() const {
    return {"Family", [](double x) { return x; }, false, false, false, [](double x) { return x; }};
}

void Family::vi_change_param(uint8_t index, double value) {}

double Family::vi_return_param(uint8_t index) const {
    return 0.0;
}

// Private functions ---------------------------------------------------------------------------------------------------

double Family::_ilogit(double x) {
    return 1 / (1 + exp(-x));
}

double Family::_logit(double x) {
    return log(x) - log(1 - x);
}

std::function<double(double)> Family::_transform_define(const std::string& transform) {
    if (transform == TRANSFORM_TANH)
        return [](double x) { return tanh(x); };
    else if (transform == TRANSFORM_EXP)
        return [](double x) { return exp(x); };
    else if (transform == TRANSFORM_LOGIT)
        return [](double x) { return _ilogit(x); };
    else if (transform.empty())
        return [](double x) { return x; };
    else
        return nullptr;
}

std::function<double(double)> Family::_itransform_define(const std::string& transform) {
    if (transform == TRANSFORM_TANH)
        return [](double x) { return atanh(x); };
    else if (transform == TRANSFORM_EXP)
        return [](double x) { return log(x); };
    else if (transform == TRANSFORM_LOGIT)
        return [](double x) { return _logit(x); };
    else if (transform.empty())
        return [](double x) { return x; };
    else
        return nullptr;
}

std::string Family::_itransform_name_define(const std::string& transform) {
    if (transform == TRANSFORM_TANH)
        return ITRANSFORM_TANH;
    else if (transform == TRANSFORM_EXP)
        return ITRANSFORM_EXP;
    else if (transform == TRANSFORM_LOGIT)
        return ITRANSFORM_LOGIT;
    // else if (transform.empty())
    //     return "";
    else
        return ""; // "None" in Python
}
