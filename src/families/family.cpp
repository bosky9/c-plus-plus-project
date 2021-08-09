#include <cmath>;
#include <functional>;

class Family {
    char* _transform_name;
    std::function<double (double)> _transform;
    char* _itransform_name;
    std::function<double (double)> _itransform;
    
    Family(char* transform="") : _transform_name(transform) {
        _transform = transform_define(transform);
        _itransform = itransform_define(transform);
        _itransform_name = itransform_name_define(transform);
    }

    static double logit(double x) {
        return log(x) - log(1 - x);
    }

    static double ilogit(double x) {
        return 1 / (1 + exp(-x));
    }

    static std::function<double (double)> transform_define(char* transform) {
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

    static std::function<double (double)> itransform_define(char* transform) {
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

    static char* itransform_name_define(char* transform) {
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
};