#pragma once

#include <list>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "covariances.hpp"
#include "families/family.hpp"
#include "headers.hpp"
#include "matplotlibcpp.hpp"
#include "multivariate_normal.hpp"
#include "output/tableprinter.hpp"

namespace plt = matplotlibcpp;

/**
 * @brief Class that represents a latent variable
 */
class LatentVariable final {
private:
    std::string _name;                        ///< Name of the latent variable
    size_t _index;                            ///< Index of the latent variable
    std::unique_ptr<Family> _prior;           ///< The prior for the latent variable, e.g. Normal(0,1)
    std::function<double(double)> _transform; ///< The transform function of the prior
    double _start;                            ///< Starting value
    std::unique_ptr<Family> _q; ///< The variational distribution for the latent variable, e.g. Normal(0,1)
    // TODO: I seguenti attributi non sono dichiarati nella classe Python ma usate in LatentVariables
    std::string _method;
    std::optional<double> _value           = std::nullopt;
    std::optional<double> _std             = std::nullopt;
    std::optional<Eigen::VectorXd> _sample = std::nullopt;

public:
    /**
     * @brief Constructor for LatentVariable
     * @param name Name of the latent variable
     * @param index Index of the latent variable
     * @param prior The prior for the latent variable, e.g. Normal(0,1)
     * @param q The variational distribution for the latent variable, e.g. Normal(0,1)
     */
    LatentVariable(std::string name, const Family& prior, const Family& q);

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
     */
    void plot_z(size_t width = 15, size_t height = 5);

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
     * @brief Set prior for the latent vairable
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
    void set_q(Family* q);
};

/**
 * @brief Class that represents a list of latent variables
 * Holds latent variable objects and contains method for latent variable manipulation. Latent variables are
 * referred to as z as shorthand. This is convention in much of the literature.
 */
class LatentVariables final {
private:
    std::string _model_name;                                         ///< Model's name
    std::vector<LatentVariable> _z_list;                             ///< List of latent variables
    std::map<std::string, std::map<std::string, size_t>> _z_indices; ///<
    bool _estimated                               = false;           ///<
    std::optional<std::string> _estimation_method = std::nullopt;    ///<

public:
    /**
     * Constructor for LatentVariables
     * @param model_name The name of the model
     */
    explicit LatentVariables(std::string model_name);

    /**
     * @brief Overload of the stream operation
     * @param stream The output stream object
     * @param latent_variables The LatentVariables object to stream
     * @return The output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const LatentVariables& latent_variables);

    /**
     * @brief Adds a latent variable
     * @param name Name of the latent variable
     * @param prior Which prior distribution? E.g. Normal(0,1)
     * @param q Which distribution to use for variational approximation
     * @param index Whether to index the variable in the z_indices dictionary
     */
    void add_z(const std::string& name, Family* prior, Family* q, bool index = true);

    /**
     * @brief Creates multiple latent variables
     * @param name Name of the latent variable
     * @param dim Dimension of the latent variable arrays
     * @param prior Which prior distribution? E.g. Normal(0,1)
     * @param q Which distribution to use for variational approximation
     */
    void create(const std::string& name, const std::vector<size_t>& dim, Family& q, Family& prior);

    /**
     * @brief Adjusts priors for the latent variables
     * @param index Which latent variable index/indices to be altered
     * @param prior Which prior distribution? E.g. Normal(0,1)
     */
    void adjust_prior(const std::vector<size_t>& index, Family& prior);

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
                std::string loc = "upper right");

    /**
     * @brief Plot samples
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void trace_plot(size_t width = 15, size_t height = 15);
};