#pragma once
#include "headers.hpp"

class Family {
public:
    std::string _transform_name;
    std::function<double (double)> _transform;
    std::string _itransform_name;
    std::function<double (double)> _itransform;
    
    Family(std::string transform = "");

private:
    static double logit(double x);
    static double ilogit(double x);

    static std::function<double (double)> transform_define(std::string transform);
    static std::function<double (double)> itransform_define(std::string transform);
    static std::string itransform_name_define(std::string transform);
};