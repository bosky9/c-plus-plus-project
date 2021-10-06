#pragma once

#include "headers.hpp"


#include <optional>

/**
 * @brief Struct for attributes returned by families
 * @details Returned by normal.setup()
 */
struct FamilyAttributes final {
    std::string name;
    std::function<double(double)> link; ///< This one is not explained in the original code
    bool scale;
    bool shape;
    bool skewness;
    std::function<double(double)> mean_transform; /**< a function which transforms the location parameter
                                                       ... in theory. In python is just np.array*/
    bool cythonized;
};

class Family {
protected:
    std::string _transform_name; ///< The name of the transform; could be exp, tanh, logit

    std::function<double(double)> _transform; /**< It saves a lambda expression,
                                                generated by the transform_define() method*/
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
     * @brief Return the transform, as a std::function - the one saved at _transform
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


    /**
     * @brief Returns the number of parameters
     * @return The number of parameters
     */
    [[nodiscard]] virtual short unsigned int get_param_no() const;

    /**
     * @brief Changes parameter at index with value specified
     * @param index Parameter's index
     * @param value New value
     */
    virtual void vi_change_param(size_t index, double value);

    /**
     * @brief Wrapper function for selecting appropriate latent variable for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @return The appropriate indexed parameter
     */
    [[nodiscard]] virtual double vi_return_param(size_t index) const;

    /**
     * @brief Get the name of the distribution family for the get_z_priors_names() method of LatentVariables
     * @return Name of the distribution family
     */
    [[nodiscard]] virtual std::string get_name() const;

    /**
     * @brief Get the description of the parameters of the distribution family for the get_z_priors_names() method of
     * LatentVariables
     * @return Description of the parameters of the distribution family
     */
    [[nodiscard]] virtual std::string get_z_name() const;

    /**
     * @brief Builds additional latent variables for this family in a probabilistic model
     * @return A list of structs (each struct contains latent variable information)
     */
    using lv_to_build = std::tuple<std::string, Family*, Family*, double>;
    virtual std::vector<lv_to_build> build_latent_variables() const;

    /**
     * @brief Draws random variables from this distribution with new latent variables
     * @param loc Location parameter for the distribution
     * @param scale Scale parameter for the distribution
     * @param shape Not actually used
     * @param skewness Not actually used
     * @param nsims Number of draws to take from the distribution
     * @return Random draws from the distribution, obtained thanks to the std::normal_distribution library.
     */
    [[nodiscard]] virtual Eigen::VectorXd draw_variable(double loc, double scale, double shape, double skewness,
                                                        int nsims);

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param size How many simulations to perform
     * @return Array of Family random variable
     */
    [[nodiscard]] virtual Eigen::VectorXd draw_variable_local(size_t size) const;

    /**
     * @brief Returns the attributes of this family if using in a probabilistic model
     * @return A struct with attributes of the family
     *
     * @details Since the attributes link, mean_transform
     *          are np.array in the original code,
     *          we could not get what their purpose was;
     *          we translated them as y = x functions.
     */
    [[nodiscard]] virtual FamilyAttributes setup() const;

    /**
     * @details Thanks to the copy costructor,
     *          this method returns a copy of the family object which calls this function.
     *          This is needed in other classes,
     *          namely LatenVariable and TSM,
     *          in order to return a deep copy of some family object.
     *
     * @return A copy of the family object which calls this function.
     */
    [[nodiscard]] virtual Family* clone() const;

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
     * @param transform Whether to apply a transformation (e.g. "exp" or "ilogit")
     *
     * @details The returned lambda functions simply calls
     *          the corresponding mathcalls.h library function.
     *
     *          If no transform function is wanted by the user,
     *          the python version allows to return a None object;
     *          here, that is converted as an y = x function (no transform applied).
     *
     *          Possible transform functions:
     *          exp, tanh, ilogit, y = x
     */
    static std::function<double(double)> transform_define(const std::string& transform);

    /**
     * @brief Define the inverse transform selected by the user
     * @details Return the inverse function associated with the transform
     * @param transform Whether to apply a transformation (e.g. "log" or "logit")
     *
     * @details The returned lambda functions simply calls
     *          the corresponding mathcalls.h library function.
     *
     *          If no transform function is wanted by the user,
     *          the python version allows to return a None object;
     *          here, that is converted as an y = x function (no transform applied).
     *
     *          Possible transform function:
     *          log, atanh, logit, y = x
     */
    static std::function<double(double)> itransform_define(const std::string& transform);

    /**
     * @brief Define any transformation performed
     * @details Used for model results table
     * @param transform Whether to apply a transformation (e.g. "exp" or "logit")
     */
    static std::string itransform_name_define(const std::string& transform);
};

using lv_to_build = std::tuple<std::string, Family*, Family*, double>;
/**<  Necessary for "build_latent_variables()" function.
 *   The python code appends to a list another list, this one:
 *   (['Normal Scale', Flat(transform='exp'), Normal(0, 3), 0.0])
 *   To translate the list above, we used this structure.
 */