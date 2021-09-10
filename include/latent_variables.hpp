#pragma once

#include "covariances.hpp"
#include "families/family.hpp"
#include "matplotlibcpp.hpp"

#include <Eigen/Core>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace plt = matplotlibcpp;

/**
 * @brief Class that represents a latent variable
 */
class LatentVariable final {
private:
    std::string _name;                        ///< Name of the latent variable
    size_t _index = 0;                        ///< Index of the latent variable
    Family _prior;                            ///< The prior for the latent variable, e.g. Normal(0,1)
    std::function<double(double)> _transform; ///< The transform function of the prior
    double _start = 0.0;
    Family _q; ///< The variational distribution for the latent variable, e.g. Normal(0,1)
    // TODO: I seguenti attributi non sono dichiarati nella classe Python ma usate in LatentVariables
    std::string _method                        = "";
    std::optional<double> _value               = std::nullopt;
    std::optional<double> _std                 = std::nullopt;
    std::optional<std::vector<double>> _sample = std::nullopt;

public:
    /**
     * @brief Constructor for LatentVariable
     * @param name Name of the latent variable
     * @param index Index of the latent variable
     * @param prior The prior for the latent variable, e.g. Normal(0,1)
     * @param q The variational distribution for the latent variable, e.g. Normal(0,1)
     */
    LatentVariable(const std::string& name, const Family& prior, const Family& q);

    /**
     * @brief Function that plots information about the latent variable
     * @param width The width of the figure to plot
     * @param height The height of the figure to plot
     */
    void plot_z(double width = 15.0, double height = 5.0);

    [[nodiscard]] std::string get_name() const;

    [[nodiscard]] Family get_prior() const;

    [[nodiscard]] std::optional<std::vector<double>> get_sample() const;

    [[nodiscard]] double get_start() const;

    [[nodiscard]] std::optional<double> get_value() const;

    [[nodiscard]] Family get_q() const;

    void set_prior(const Family& prior);

    void set_start(double start);
};

/**
 * @brief Class that represents a list of latent variables
 * Holds latent variable objects and contains method for latent variable manipulation. Latent variables are
 * referred to as z as shorthand. This is convention in much of the literature.
 */
class LatentVariables final {
private:
    std::string _model_name;
    std::vector<LatentVariable> _z_list;
    std::map<std::string, std::map<std::string, size_t>> _z_indices;
    bool _estimated                = false;
    std::string _estimation_method = "";

public:
    /**
     * Constructor for LatentVariables
     * @param model_name The name of the model
     */
    LatentVariables(const std::string& model_name);

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
    void add_z(const std::string& name, const Family& prior, const Family& q, bool index = true);

    /**
     * @brief Creates multiple latent variables
     * @param name Name of the latent variable
     * @param dim Dimension of the latent variable arrays
     * @param prior Which prior distribution? E.g. Normal(0,1)
     * @param q Which distribution to use for variational approximation
     */
    void create(const std::string& name, const std::vector<size_t>& dim, const Family& prior, const Family& q);

    /**
     * @brief Adjusts priors for the latent variables
     * @param index Which latent variable index/indices to be altered
     * @param prior Which prior distribution? E.g. Normal(0,1)
     */
    void adjust_prior(const std::vector<size_t>& index, const Family& prior);

    std::vector<std::string> get_z_names() const;

    std::vector<Family> get_z_priors() const;

    std::pair<std::vector<std::string>, std::vector<std::string>> get_z_priors_names() const;

    std::vector<std::function<double(double)>> get_z_transforms() const;

    std::vector<std::string> get_z_transforms_names() const;

    Eigen::VectorXd get_z_starting_values(bool transformed = false) const;

    Eigen::VectorXd get_z_values(bool transformed = false) const;

    std::vector<Family> get_z_approx_dist() const;

    std::vector<std::string> get_z_approx_dist_names() const;

    void set_z_values(const Eigen::VectorXd& values, const std::string& method,
                      const std::optional<Eigen::VectorXd>& std    = std::nullopt,
                      const std::optional<Eigen::VectorXd>& sample = std::nullopt);

    void set_z_starting_values(const Eigen::VectorXd& values);

    void plot_z(const std::optional<std::vector<size_t>>& indices = std::nullopt, double width = 15.0,
                double height = 5.0, int loc = 1);

    void trace_plot(double width = 15.0, double height = 15.0);
};