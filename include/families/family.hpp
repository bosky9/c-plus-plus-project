#pragma once
#include "headers.hpp"

class Family {
public:
    std::string _transform_name;
    std::function<double (double)> _transform;
    std::string _itransform_name;
    std::function<double (double)> _itransform;

    /**
     * @brief Constructor for Family
     * @param transform (string): whether to apply a transformation (e.g. "exp" or "logit")
     */
    Family(std::string transform = "");

private:
    /**
     * @brief Apply the logit transformation
     * @param x (double)
     */
    static double logit(double x);

    /**
     * @brief Apply the inverse logit transformation
     * @param x (double)
     */
    static double ilogit(double x);

    /**
     * @brief Define the transform selected by the user
     * @details Return the function associated with the transform
     * @param transform (string)
     */
    static std::function<double (double)> transform_define(std::string transform);

    /**
     * @brief Define the transform selected by the user with its inverse
     * @details Return the inverse function associated with the transform
     * @param transform (string)
     */
    static std::function<double (double)> itransform_define(std::string transform);

    /**
     * @brief Define any transformation performed
     * @details Used for model results table
     * @param transform (string)
     */
    static std::string itransform_name_define(std::string transform);
};