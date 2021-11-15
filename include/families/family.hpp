/**
 * @file family.hpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd

#include <functional> // std::function
#include <memory>     // std::unique_ptr
#include <string>     // std::string

/**
 * @struct FamilyAttributes family.hpp
 * @brief Struct for attributes returned by families
 * @details Returned by normal.setup()
 */
struct FamilyAttributes final {
    std::string name;
    std::function<double(double)> link; ///< This one is not explained in the original code
    bool scale;
    bool shape;
    bool skewness;
    std::function<double(double)>
            mean_transform; /**< A function which transforms the location parameter in theory. In Python is just a
                               np.array, a function which returns the same passed value.*/
};

/**
 * @class Family family.hpp
 * @brief Generic class for families of distributions
 */
class Family {
public:
    /**
     * @brief Constructor for Family
     * @param transform Whether to apply a transformation (e.g. "exp" or "_logit")
     */
    explicit Family(const std::string& transform = "");

    /**
     * @brief Check if Family objects are equal
     * @param family1 First object
     * @param family2 Second object
     * @return If the two objects are equal
     */
    friend bool operator==(const Family& family1, const Family& family2);

    // Get methods -----------------------------------------------------------------------------------------------------

    /**
     * @brief Get the name of the distribution family for the get_z_priors_names() method of LatentVariables
     * @return Name of the distribution family
     */
    [[nodiscard]] virtual std::string get_name() const;

    /**
     * @brief Return the inverse transform
     * @return The inverse transform
     */
    [[nodiscard]] std::function<double(double)> get_itransform() const;

    /**
     * @brief Return the name of the inverse transform
     * @return The name of the inverse transform
     */
    [[nodiscard]] std::string get_itransform_name() const;

    /**
     * @brief Return the transform, as a std::function - the one saved at _transform
     * @return The transform
     */
    [[nodiscard]] std::function<double(double)> get_transform() const;

    /**
     * @brief Return the name of the transform
     * @return The name of the transform
     */
    [[nodiscard]] std::string get_transform_name() const;

    // Virtual functions for subclasses --------------------------------------------------------------------------------

    /**
     * @brief Builds additional latent variables for this family in a probabilistic model
     * @return A list of tuples (each tuple contains latent variable information)
     */
    using lv_to_build = std::tuple<std::string, Family*, Family*, double>;
    [[nodiscard]] virtual std::vector<lv_to_build> build_latent_variables() const;

    /**
     * @brief Returns a clone of the current object
     * @return A copy of the family object which calls this function
     *
     * @details Thanks to the copy costructor, this method returns a (pointer to a) copy of the family object which
     *          calls this function. This is needed in other classes, namely LatenVariable and TSM, in order to return a
     *          deep copy of some family object.
     */
    [[nodiscard]] virtual std::unique_ptr<Family> clone() const;

    /**
     * @brief Draws random variables from this distribution with new latent variables
     * @param loc Location parameter(s) for the distribution
     * @param scale Scale parameter for the distribution
     * @param nsims Number of draws to take from the distribution
     * @return Random draws from the distribution, obtained thanks to the std::normal_distribution library.
     * @details Parameters shape and skewness were present but not actually used
     */
    [[nodiscard]] virtual Eigen::VectorXd draw_variable(double loc, double scale, int64_t nsims) const;
    [[nodiscard]] virtual Eigen::VectorXd draw_variable(const Eigen::VectorXd& loc, double scale, int64_t nsims) const;

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param size How many simulations to perform
     * @return Array of Family random variable
     */
    [[nodiscard]] virtual Eigen::VectorXd draw_variable_local(int64_t size) const;

    /**
     * @brief Returns the number of parameters
     * @return The number of parameters
     */
    [[nodiscard]] virtual uint8_t get_param_no() const;

    /**
     * @brief Get the description of the parameters of the distribution family for the get_z_priors_names() method of
     * LatentVariables
     * @return Description of the parameters of the distribution family
     */
    [[nodiscard]] virtual std::string get_z_name() const;

    /**
     * @brief Log PDF for generic Family prior
     * @param mu Latent variable for which the prior is being formed over
     * @return log(p(mu))
     */
    [[nodiscard]] virtual double logpdf(double mu) const;

    /**
     * @brief Negative loglikelihood function
     * @param y Univariate time series
     * @param mean Array of location parameters for the distribution
     * @param scale Scale parameter for the distribution
     * @return Negative loglikelihood of the family
     *
     * @details Parameters shape and skewness were present but not actually used.
     */
    [[nodiscard]] virtual double neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean,
                                                   double scale) const;

    /**
     * @brief Returns the attributes of this family if using in a probabilistic model
     * @return A struct with attributes of the family
     *
     * @details Since the attributes link, mean_transform are np.array in the original code, we could not get what their
     *          purpose was; we translated them as y = x functions.
     */
    [[nodiscard]] virtual FamilyAttributes setup() const;

    /**
     * @brief Changes parameter at index with value specified
     * @param index Parameter's index
     * @param value New value
     */
    virtual void vi_change_param(uint8_t index, double value);

    /**
     * @brief Wrapper function for selecting appropriate latent variable for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @return The appropriate indexed parameter
     */
    [[nodiscard]] virtual double vi_return_param(uint8_t index) const;

protected:
    std::string _transform_name;               ///< The name of the transform; could be exp, tanh, logit
    std::function<double(double)> _transform;  ///< Transform function: it saves a lambda expression, generated by the
                                               ///< _transform_define() method
    std::string _itransform_name;              ///< The name of the inverse transform; could be log, arctanh, ilogit
    std::function<double(double)> _itransform; ///< Inverse transform function

    static const std::string TRANSFORM_EXP;
    static const std::string TRANSFORM_TANH;
    static const std::string TRANSFORM_LOGIT;
    static const std::string ITRANSFORM_EXP;
    static const std::string ITRANSFORM_TANH;
    static const std::string ITRANSFORM_LOGIT;

private:
    /**
     * @brief Apply the inverse _logit transformation
     * @param x
     */
    static double _ilogit(double x);

    /**
     * @brief Apply the _logit transformation
     * @param x
     */
    static double _logit(double x);

    /**
     * @brief Define the transform selected by the user
     * @details Return the function associated with the transform
     * @param transform Whether to apply a transformation (e.g. "exp" or "_ilogit")
     *
     * @details The returned lambda functions simply calls the corresponding mathcalls.h library function.
     *          If no transform function is wanted by the user, the Python version allows to return a None object;here,
     *          that is converted as an y = x function (no transform applied).
     *          Possible transform functions: exp, tanh, _ilogit, y = x
     */
    static std::function<double(double)> _transform_define(const std::string& transform);

    /**
     * @brief Define the inverse transform selected by the user
     * @details Return the inverse function associated with the transform
     * @param transform Whether to apply a transformation (e.g. "log" or "_logit")
     *
     * @details The returned lambda functions simply calls the corresponding mathcalls.h library function.
     *          If no transform function is wanted by the user, the Python version allows to return a None object;here,
     *          that is converted as an y = x function (no transform applied).
     *          Possible transform functions: exp, tanh, _ilogit, y = x
     */
    static std::function<double(double)> _itransform_define(const std::string& transform);

    /**
     * @brief Define any transformation performed
     * @details Used for model results table
     * @param transform Whether to apply a transformation (e.g. "exp" or "_logit")
     */
    static std::string _itransform_name_define(const std::string& transform);
};

using lv_to_build = std::tuple<std::string, Family*, Family*, double>;
/**<  Necessary for "build_latent_variables()" function.
 *   The Python code appends to a list another list, this one:
 *   (['Normal Scale', Flat(transform='exp'), Normal(0, 3), 0.0])
 *   To translate the list above, we used this tuple.
 */