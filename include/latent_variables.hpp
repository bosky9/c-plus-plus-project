#pragma once

#include "covariances.hpp"
#include "families/family.hpp"
#include "headers.hpp"
#include "matplotlibcpp.hpp"
#include "multivariate_normal.hpp"
#include "output/tableprinter.hpp"

#include <list>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace plt = matplotlibcpp;

/**
 * @class LatentVariable latent_variables.hpp
 * @brief Class that represents a latent variable
 *
 * @details Using a unique_pointer for _prior and _q
 *          allows, inside the constructor,
 *          the passage of every subclass of Family.
 */
class LatentVariable final {

public:
    /**
     * @brief Constructor for LatentVariable
     * @param name Name of the latent variable
     * @param prior The prior for the latent variable, e.g. Normal(0,1)
     * @param q The variational distribution for the latent variable, e.g. Normal(0,1)
     */
    LatentVariable(const std::string& name, const Family& prior, const Family& q);

    /**
     * @brief Copy constructor
     * @param lv The LatentVariable object to copy
     */
    LatentVariable(const LatentVariable& lv);

    /**
     * @brief Move constructor
     * @param lv The LatentVariable object to move
     */
    LatentVariable(LatentVariable&& lv) noexcept;

    /**
     * @brief Assignment operator
     * @param lv The LatentVariable object to assign
     * @return The self modified object
     */
    LatentVariable& operator=(const LatentVariable& lv);

    /**
     * @brief Move assignment operator
     * @param lv The LatentVariable object to move
     * @return The self modified object
     */
    LatentVariable& operator=(LatentVariable&& lv) noexcept;

    /**
     * @brief Destructor
     */
    ~LatentVariable();

    /**
     * @brief Function that plots information about the latent variable
     * @param width The width of the figure to plot
     * @param height The height of the figure to plot
     *
     * @detail This one uses the "matplotlibcpp.hpp" library, which calls Python to print a matplot.
     */
    void plot_z(size_t width = 15, size_t height = 5) const;

    /**
     * @brief Returns the method's name
     * @return Method
     */
    [[nodiscard]] std::string get_method() const;

    /**
     * @brief Returns the name of the latent variable
     * @return Name
     */
    [[nodiscard]] std::string get_name() const;

    /**
     * @brief Returns the prior for the latent variable
     * @return Prior
     */
    [[nodiscard]] Family* get_prior() const;

    /**
     * @brief Returns the sample (optional)
     * @return Sample
     */
    [[nodiscard]] std::optional<Eigen::VectorXd> get_sample() const;

    /**
     * @brief Returns start
     * @return Start
     */
    [[nodiscard]] double get_start() const;

    /**
     * @brief Returns STD (optional)
     * @return STD
     */
    [[nodiscard]] std::optional<double> get_std() const;

    /**
     * @brief Returns value
     * @return Value
     */
    [[nodiscard]] std::optional<double> get_value() const;

    /**
     * @brief Returns the variational distribution for the latent variable
     * @return Variational distribution
     */
    [[nodiscard]] Family* get_q() const;

    /**
     * @brief Set prior for the latent variable
     * @param prior Prior
     */
    void set_prior(const Family& prior);

    /**
     * @brief Set start
     * @param start Start
     */
    void set_start(double start);

    /**
     * @brief Set method's name
     * @param method Method name
     */
    void set_method(const std::string& method);

    /**
     * @brief Set value
     * @param value Value
     */
    void set_value(double value);

    /**
     * @brief Set STD
     * @param std STD
     */
    void set_std(double std);

    /**
     * @brief Set sample
     * @param sample Sample
     */
    void set_sample(const Eigen::VectorXd& sample);

    /**
     * @brief Sets a new family object in q
     * @param family Family object
     */
    void set_q(const Family& q);

private:
    std::string _name;                                     ///< Name of the latent variable
    size_t _index;                                         ///< Index of the latent variable
    std::unique_ptr<Family> _prior;                        ///< The prior for the latent variable, e.g. Normal(0,1)
    std::function<double(double)> _transform;              ///< The transform function of the prior
    double _start;                                         ///< Starting value
    std::unique_ptr<Family> _q;                            ///< The variational distribution for the latent variable, e.g. Normal(0,1)
    // The following attributes aren't declared in Python code but used in LatentVariables class
    std::string _method;                                   ///< The estimation method
    std::optional<double> _value           = std::nullopt; ///< The value of the latent variable
    std::optional<double> _std             = std::nullopt; ///< The standard deviation of the latent variable
    std::optional<Eigen::VectorXd> _sample = std::nullopt; ///< The sample values of the latent variables
};

/**
 * @class LatentVariables latent_variables.hpp
 * @brief Class that represents a list of latent variables.
 * Holds latent variable objects and contains method for latent variable manipulation. Latent variables are
 * referred to as z as shorthand. This is convention in much of the literature.
 */
class LatentVariables final {

public:
    /**
     * Constructor for LatentVariables
     * @param model_name The name of the model
     */
    explicit LatentVariables(const std::string& model_name);

    /**
     * @brief Overload of the stream operation
     * @param stream The output stream object
     * @param latent_variables The LatentVariables object to stream
     * @return The output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const LatentVariables& latent_variables);

    /**
     * @brief Appends a latent variable to the _z_list array
     * @param name Name of the latent variable
     * @param prior Which prior distribution? E.g. Normal(0,1)
     * @param q Which distribution to use for variational approximation
     * @param index Whether to index the variable in the z_indices dictionary
     */
    void add_z(const std::string& name, const Family& prior, const Family& q, bool index = true);

    /**
     * @brief Creates multiple latent variables
     * @param name Name of the latent variable
     * @param dim Dimension of the latent variable arrays
     * @param prior Which prior distribution? E.g. Normal(0,1)
     * @param q Which distribution to use for variational approximation
     *
     * @detail  In Python, there is a recursive function which,
     *          given a list [x,y,...,z] e.g. [3,2],
     *          it returns  "(1,1)" "(1,2)" , "(2,1)" , "(2,2)",
     *                      "(3,1)", "(3,2)"
     *
     *          The number of elements is:
     *              x*y*...*z
     *          6 lists in the e.g.
     *
     *          For the first position, there are two 1 two 2 two 3.
     *          Basically, it is, upper approximated,
     *          given that (6/3 = 2), -> (index / (6/3))
     *          since we want to split the whole set in three parts,
     *          (1/2 = 1, ...)
     *          (2/2 = 1, ...)
     *          (3/2 = 2, ...)
     *          (4/2 = 2, ...)
     *          (5/2 = 3, ...)
     *          (6/2 = 3, ...)
     *
     *          For the second position, each first position has either another 0 or 1;
     *          We are now dealing with 3 sets of two components,
     *          and our current number is [-,2];
     *          -> (index / (2 / 2))
     *          so this mean
     *          (1, 1/1 = 1)
     *          (1, 2/1 = 2)
     *          (2, 1/1 = 1)
     *          (2, 2/1 = 2)
     *          (3, 1/1 = 1)
     *          (3, 2/1 = 2)
     *
     *          Please notice that at each position x(i) the number by which we divide is
     *          the number necessary to split the set in x(i) parts.
     *          This can both be obtained by division with (number_of_current_elements / x(i)),
     *          or by multiplication of x(i+1) * x(i+2) * ... * 1)
     *
     *          While the index ranges from 1 to number_of_current_elements,
     *          which is (previous num of indexes / previous x(i)).
     *          At position 0 -> [6 (total) / 1] = 6, init
     *          At position 1 -> [6 (previous value)/ 3  (previous num of dims)] = 2
     */
    void create(const std::string& name, const std::vector<size_t>& dim, const Family& q, const Family& prior);

    /**
     * @brief Adjusts priors for the latent variables in _z_list
     * @param index Which latent variable index/indices to be altered
     * @param prior Which prior distribution? E.g. Normal(0,1)
     */
    void adjust_prior(const std::vector<int64_t>& index, const Family& prior);

    /**
     * @brief Returns list of LatentVariable objects
     * @return List of LatentVariable objects
     */
    [[nodiscard]] std::vector<LatentVariable> get_z_list() const;

    /**
     * @brief Returns latent variables' names
     * @return Latent variables' names
     */
    [[nodiscard]] std::vector<std::string> get_z_names() const;

    /**
     * @brief Returns latent variables' priors
     * @return Latent variables' priors
     */
    [[nodiscard]] std::vector<Family*> get_z_priors() const;

    /**
     * @brief Returns latent variables' priors' names
     * @return Latent variables' priors' names
     */
    [[nodiscard]] std::pair<std::vector<std::string>, std::vector<std::string>> get_z_priors_names() const;

    /**
     * @brief Returns latent variables' transforms
     * @return Latent variables' transforms
     */
    [[nodiscard]] std::vector<std::function<double(double)>> get_z_transforms() const;

    /**
     * @brief Returns latent variables' transforms' names
     * @return Latent variables' transforms' names
     */
    [[nodiscard]] std::vector<std::string> get_z_transforms_names() const;

    /**
     * @brief Returns latent variables' starting values
     * @param transformed If values need to be transformed before
     * @return Latent variablesì starting values
     */
    [[nodiscard]] Eigen::VectorXd get_z_starting_values(bool transformed = false) const;

    /**
     * @brief Returns latent variables' values
     * @param transformed If values need to be transformed before
     * @return  Latent variables' values
     */
    [[nodiscard]] Eigen::VectorXd get_z_values(bool transformed = false) const;

    /**
     * @brief Returns the approximate distributions of the latent variables
     * @return Approximate distributions
     */
    [[nodiscard]] std::vector<Family*> get_z_approx_dist() const;

    /**
     * @brief Returns the approximate distributions' names of the latent variables
     * @return Approximate distributions' names
     */
    [[nodiscard]] std::vector<std::string> get_z_approx_dist_names() const;

    /**
     * @brief Checks if latent variables are estimated
     * @return If latent variables are estimated
     */
    [[nodiscard]] bool is_estimated() const;

    /**
     * @brief Get the estimation method if setted
     * @return The estimation method if setted, std::nullopt_t otherwise
     */
    [[nodiscard]] std::optional<std::string> get_estimation_method() const;

    /**
     * @brief Sets estimation method
     * @param method Estimation method
     */
    void set_estimation_method(const std::string& method);

    /**
     * @brief Set values to latent variables
     * @param values Vector of values to set
     * @param method Method name
     * @param stds Vector of STDs
     * @param samples Vector of samples
     */
    void set_z_values(const Eigen::VectorXd& values, const std::string& method,
                      const std::optional<Eigen::VectorXd>& stds    = std::nullopt,
                      const std::optional<Eigen::MatrixXd>& samples = std::nullopt);

    /**
     * @brief Set starting values to latent variables
     * @param values Vector of starting values to set
     */
    void set_z_starting_values(const Eigen::VectorXd& values);

    /**
     * @brief Set starting value to a latent variable
     * @param index The index of the latent variable in _z_list
     * @param value Starting value to set
     */
    void set_z_starting_value(size_t index, double value);

    /**
     * @brief Plots the latent variables
     * @param indices Vector of indices to plot
     * @param width Width of the figure
     * @param height Height of the figure
     * @param loc Location of the legend
     */
    void plot_z(const std::optional<std::vector<size_t>>& indices = std::nullopt, size_t width = 15, size_t height = 5,
                const std::string& loc = "upper right") const;

    /**
     * @brief Plot samples
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void trace_plot(size_t width = 15, size_t height = 15);

private:
    std::string _model_name;                                         ///< Model's name
    std::vector<LatentVariable> _z_list;                             ///< List of latent variables, as a std::vector
    std::map<std::string, std::map<std::string, size_t>> _z_indices; ///< Info about latent variables
    bool _estimated                               = false;           ///< Status of the latent variables estimation
    std::optional<std::string> _estimation_method = std::nullopt;    ///< Method of the latent variables estimation
};