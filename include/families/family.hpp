#pragma once
#include "headers.hpp"

class Family {
protected:
    std::string _transform_name;
    std::function<double(double)> _transform;
    std::string _itransform_name;
    std::function<double(double)> _itransform;

public:
    static const std::string TRANSFORM_EXP;
    static const std::string TRANSFORM_TANH;
    static const std::string TRANSFORM_LOGIT;

    /**
     * @brief Constructor for Family
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    explicit Family(const std::string& transform = "");

    /**
     * @brief Copy constructor for Family
     * @param family A Family object
     */
    Family(const Family& family);

    /**
     * @brief Move constructor for Family
     * @param family A Family object
     */
    Family(Family&& family) noexcept;

    /**
     * @brief Assignment operator for Family
     * @param family A Family object
     */
    Family& operator=(const Family& family);

    /**
     * @brief Move assignment operator for Family
     * @param family A Family object
     */
    Family& operator=(Family&& family) noexcept;

    /**
     * @brief Check if Family objects are equal
     * @param family1 First object
     * @param family2 Second object
     * @return If the two objects are equal
     */
    friend bool is_equal(const Family& family1, const Family& family2);

    /**
     * @brief Return the name of the transform
     * @return The name of the transform
     */
    [[nodiscard]] std::string get_transform_name() const;

    /**
     * @brief Return the transform
     * @return The transform
     */
    [[nodiscard]] std::function<double(double)> get_transform() const;

    /**
     * @brief Return the name of the inverse transform
     * @return The name of the inverse transform
     */
    [[nodiscard]] std::string get_itransform_name() const;

    /**
     * @brief Return the inverse transform
     * @return The inverse transform
     */
    [[nodiscard]] std::function<double(double)> get_itransform() const;

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
    static std::function<double(double)> transform_define(const std::string& transform);

    /**
     * @brief Define the transform selected by the user with its inverse
     * @details Return the inverse function associated with the transform
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    static std::function<double(double)> itransform_define(const std::string& transform);

    /**
     * @brief Define any transformation performed
     * @details Used for model results table
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    static std::string itransform_name_define(const std::string& transform);
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