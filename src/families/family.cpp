#include "families/family.hpp"

const std::string Family::TRANSFORM_EXP   = "exp";
const std::string Family::TRANSFORM_TANH  = "tanh";
const std::string Family::TRANSFORM_LOGIT = "logit";

Family::Family(const std::string& transform) : _transform_name{transform} {
    _transform       = transform_define(transform);
    _itransform      = itransform_define(transform);
    _itransform_name = itransform_name_define(transform);
}

Family::Family(const Family& family) {
    _transform_name  = family._transform_name;
    _transform       = family._transform;
    _itransform_name = family._itransform_name;
    _itransform      = family._itransform;
}

Family::Family(Family&& family) noexcept {
    _transform_name  = family._transform_name;
    _transform       = family._transform;
    _itransform_name = family._itransform_name;
    _itransform      = family._itransform;
    family._transform_name.clear();
    family._transform = nullptr;
    family._itransform_name.clear();
    family._itransform = nullptr;
}

Family& Family::operator=(const Family& family) {
    if (this == &family)
        return *this;
    _transform_name  = family._transform_name;
    _transform       = family._transform;
    _itransform_name = family._itransform_name;
    _itransform      = family._itransform;
    return *this;
}

Family& Family::operator=(Family&& family) noexcept {
    _transform_name  = family._transform_name;
    _transform       = family._transform;
    _itransform_name = family._itransform_name;
    _itransform      = family._itransform;
    family._transform_name.clear();
    family._transform = nullptr;
    family._itransform_name.clear();
    family._itransform = nullptr;
    return *this;
}

bool is_equal(const Family& family1, const Family& family2) {
    return family1._transform_name == family2._transform_name && family1._itransform_name == family2._itransform_name;
}

double Family::logit(double x) {
    return log(x) - log(1 - x);
}

double Family::ilogit(double x) {
    return 1 / (1 + exp(-x));
}

std::function<double(double)> Family::transform_define(const std::string& transform) {
    if (transform == "tanh")
        return [](double x) { return tanh(x); };
    else if (transform == "exp")
        return [](double x) { return exp(x); };
    else if (transform == "logit")
        return [](double x) { return ilogit(x); };
    else if (transform.empty())
        return [](double x) { return x; };
    //@TODO: Ritornare la funzione costante come sopra? O nullptr?
    else
        return [](double x) { return x; };
}

std::function<double(double)> Family::itransform_define(const std::string& transform) {
    if (transform == "tanh")
        return [](double x) { return atanh(x); };
    else if (transform == "exp")
        return [](double x) { return log(x); };
    else if (transform == "logit")
        return [](double x) { return logit(x); };
    else if (transform.empty())
        return [](double x) { return x; };
    //@TODO: Ritornare la funzione costante come sopra? O nullptr?
    else
        return [](double x) { return x; };
}

std::string Family::itransform_name_define(const std::string& transform) {
    if (transform == "tanh")
        return "arctanh";
    else if (transform == "exp")
        return "log";
    else if (transform == "logit")
        return "ilogit";
    else if (transform.empty())
        return "";
    // TODO: Ritornare NULL o stringa vuota?
    else
        return "";
}

std::string Family::get_transform_name() const {
    return _transform_name;
}

std::function<double(double)> Family::get_transform() const {
    return _transform;
}

std::string Family::get_itransform_name() const {
    return _itransform_name;
}

std::function<double(double)> Family::get_itransform() const {
    return _itransform;
}

std::string Family::get_name() const {
    return "Prior distribution not detected";
}

std::string Family::get_z_name() const {
    return "";
}

Eigen::VectorXd Family::draw_variable_local(size_t size) const {
    return {};
}

void Family::vi_change_param(size_t index, double value) {}

double Family::vi_return_param(size_t index) const {
    return 0.0;
}

short unsigned int Family::get_param_no() const {
    return 0;
}


Family* Family::clone() const {
    return new Family(*this);
}