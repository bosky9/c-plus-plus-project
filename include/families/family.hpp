#pragma once
#include "headers.hpp"

class Family {
public:
    std::string _transform_name;
    std::function<double(double)> _transform;
    std::string _itransform_name;
    std::function<double(double)> _itransform;

    /**
     * @brief Constructor for Family
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    Family(const std::string &transform = "");

private:
    /**
     * @brief Apply the logit transformation
     * @param x
     */
    static double logit(double x);

    /**
     * @brief Apply the inverse logit transformation
     * @param x
     */
    static double ilogit(double x);

    /**
     * @brief Define the transform selected by the user
     * @details Return the function associated with the transform
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    static std::function<double(double)> transform_define(const std::string &transform);

    /**
     * @brief Define the transform selected by the user with its inverse
     * @details Return the inverse function associated with the transform
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    static std::function<double(double)> itransform_define(const std::string &transform);

    /**
     * @brief Define any transformation performed
     * @details Used for model results table
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    static std::string itransform_name_define(const std::string &transform);
};

/**
 * @brief Struct for attributes returned by families
 */
struct FamilyAttributes {
    std::string name;
    std::function<double(double)> link;
    bool scale;
    bool shape;
    bool skewness;
    std::function<double(double)> mean_transform;
    bool cythonized;
};