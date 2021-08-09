#pragma once
#include "../headers.hpp";

class Family {
public:
    char* _transform_name;
    std::function<double (double)> _transform;
    char* _itransform_name;
    std::function<double (double)> _itransform;
private:
    Family(char* transform="");
    static double logit(double x);
    static double ilogit(double x);

    static std::function<double (double)> transform_define(char* transform);
    static std::function<double (double)> itransform_define(char* transform);
    static char* itransform_name_define(char* transform);
};