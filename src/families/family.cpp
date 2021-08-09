#include "../../include/families/family.hpp";



Family::Family(char* transform="") : _transform_name(transform) {
    _transform = transform_define(transform);
    _itransform = itransform_define(transform);
    _itransform_name = itransform_name_define(transform);
}

double Family::logit(double x) {
    return log(x) - log(1 - x);
}

double Family::ilogit(double x) {
    return 1 / (1 + exp(-x));
}

std::function<double (double)> Family::transform_define(char* transform) {
    if (transform == "tanh")
        return [](double x){ return tanh(x); };
    else if (transform == "exp")
        return [](double x){ return exp(x); };
    else if (transform == "logit")
        return [](double x){ return ilogit(x); };
    else if (transform == "")
        return [](double x){ return x; };
    else 
        return NULL;
}

std::function<double (double)> Family::itransform_define(char* transform) {
    if (transform == "tanh")
        return [](double x){ return atanh(x); };
    else if (transform == "exp")
        return [](double x){ return log(x); };
    else if (transform == "logit")
        return [](double x){ return logit(x); };
    else if (transform == "")
        return [](double x){ return x; };
    else 
        return NULL;
}

char* Family::itransform_name_define(char* transform) {
    if (transform == "tanh")
        return "arctanh";
    else if (transform == "exp")
        return "log";
    else if (transform == "logit")
        return "ilogit";
    else if (transform == "")
        return "";
    else
        return NULL;
}
