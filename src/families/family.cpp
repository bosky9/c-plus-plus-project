#include "families/family.hpp"

Family::Family(const std::string &transform) : _transform_name{transform} {
    _transform       = transform_define(transform);
    _itransform      = itransform_define(transform);
    _itransform_name = itransform_name_define(transform);
}

double Family::logit(double x) {
    return log(x) - log(1 - x);
}

double Family::ilogit(double x) {
    return 1 / (1 + exp(-x));
}

std::function<double(double)> Family::transform_define(const std::string &transform) {
    if (transform == "tanh")
        return [](double x) { return tanh(x); };
    else if (transform == "exp")
        return [](double x) { return exp(x); };
    else if (transform == "logit")
        return [](double x) { return ilogit(x); };
    else if (transform.empty())
        return [](double x) { return x; };
    else
        return nullptr;
}

std::function<double(double)> Family::itransform_define(const std::string &transform) {
    if (transform == "tanh")
        return [](double x) { return atanh(x); };
    else if (transform == "exp")
        return [](double x) { return log(x); };
    else if (transform == "logit")
        return [](double x) { return logit(x); };
    else if (transform.empty())
        return [](double x) { return x; };
    else
        return nullptr;
}

std::string Family::itransform_name_define(const std::string &transform) {
    if (transform == "tanh")
        return "arctanh";
    else if (transform == "exp")
        return "log";
    else if (transform == "logit")
        return "ilogit";
    else if (transform.empty())
        return "";
    else
        return NULL;
}
