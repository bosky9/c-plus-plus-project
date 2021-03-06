/**
 * @file results.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core"           // Eigen::VectorXd, Eigen::MatrixXd
#include "latent_variables.hpp" // LatentVariables

#include <functional> // std::function
#include <optional>   // std::optional, std::nullopt
#include <ostream>    // std::ostream
#include <string>     // std::string
#include <vector>     // std::vector

/**
 * @class Results results.hpp
 */
class Results {
public:
    virtual void summary(bool transformed) = 0;

    /**
     * @detail  Since we will be using Results* pointers
     *          inside TSM to refer to Results subclasses,
     *          a virtual destructor is necessary to avoid undefined behaviour.
     */
    virtual ~Results() = default;

    /**
     * @brief Returns latent variables
     * @return Latent variables in _z
     */
    [[nodiscard]] LatentVariables get_z() const;

protected:
    std::vector<std::string> _x_names;
    std::string _model_name;
    std::string _model_type;
    LatentVariables _z;
    Eigen::VectorXd _z_values;
    Eigen::MatrixXd _data; ///< Predicted values for the time series and length-adjusted time series
    std::vector<double> _index;
    bool _multivariate_model;
    std::function<double(Eigen::VectorXd)> _objective_object; ///< Likelihood or posterior
    std::string _method;
    uint8_t _z_hide;
    int _max_lag;
    Eigen::VectorXd _signal;
    // The variables _scores, _states and _states_var aren't used with ARMA models
    std::optional<Eigen::VectorXd> _scores;
    std::optional<Eigen::VectorXd> _states;
    std::optional<Eigen::VectorXd> _states_var;
    size_t _data_length;
    std::string _data_name;
    uint8_t _rounding_points;
    double _aic;
    double _bic;

    /**
     * @brief Constructor for Results
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param results
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param scores
     * @param states
     * @param states_var
     */
    Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
            const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
            std::vector<double> index, bool multivariate_model, std::function<double(Eigen::VectorXd)> objective_object,
            std::string method, bool z_hide, int max_lag, Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
            std::optional<Eigen::VectorXd> scores = std::nullopt, std::optional<Eigen::VectorXd> states = std::nullopt,
            std::optional<Eigen::VectorXd> states_var = std::nullopt);

    /**
     * @brief Rounds given specific rounding points
     * @param x Value to round
     * @param rounding_points Rounding points
     * @return Value rounded up to rounding points
     */
    [[nodiscard]] static double round_to(double x, uint8_t rounding_points);
};

/**
 * @class MLEResults results.hpp
 * @details Inheritance declaration has to be public for pointers (Return* p = &MLEResults{...})
 */
class MLEResults : public Results {
public:
    /**
     * @brief Returns the Inverse Hessian matrix
     * @return Inverse Hessian matrix
     */
    [[nodiscard]] Eigen::MatrixXd get_ihessian() const;

    /**
     * @brief Constructor for MLEResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param results
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param scores
     * @param states
     * @param states_var
     */
    MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
               const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
               Eigen::MatrixXd data, std::vector<double> index, bool multivariate_model,
               std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
               Eigen::MatrixXd ihessian = Eigen::VectorXd::Zero(0), Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
               std::optional<Eigen::VectorXd> scores     = std::nullopt,
               std::optional<Eigen::VectorXd> states     = std::nullopt,
               std::optional<Eigen::VectorXd> states_var = std::nullopt);

    /**
     * @brief Stream operator for MLEResults
     * @param stream Output stream
     * @param mle_results MLEResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MLEResults& results);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;

private:
    Eigen::VectorXd _results;
    Eigen::MatrixXd _ihessian;
    double _loglik;

    /**
     * @brief Prints results with hessian
     * @param transformed
     */
    void summary_with_hessian(bool transformed = true) const;

    /**
     * @brief Prints results without hessian
     */
    void summary_without_hessian() const;
};

/**
 * @class BBVIResults results.hpp
 */
class BBVIResults : public Results {
public:
    /**
     * @brief Constructor for BBVIResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ses
     * @param signal
     * @param scores
     * @param elbo_records
     * @param states
     * @param states_var
     */
    BBVIResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                std::vector<double> index, bool multivariate_model,
                std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
                Eigen::VectorXd ses, Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
                std::optional<Eigen::VectorXd> scores     = std::nullopt,
                Eigen::VectorXd elbo_records              = Eigen::VectorXd::Zero(0),
                std::optional<Eigen::VectorXd> states     = std::nullopt,
                std::optional<Eigen::VectorXd> states_var = std::nullopt);

    /**
     * @brief Stream operator for BBVIResults
     * @param stream Output stream
     * @param results BBVIResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const BBVIResults& results);

    /**
     * @brief Plots the ELBO progress (if present)
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_elbo(size_t width = 15, size_t height = 7) const;

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;

private:
    Eigen::MatrixXd _ihessian;
    Eigen::VectorXd _ses;
    Eigen::VectorXd _elbo_records;
    Eigen::MatrixXd _chain;          ///< Chains for each parameter
    Eigen::VectorXd _mean_est;       ///< Mean values for each parameter
    Eigen::VectorXd _median_est;     ///< Median values for each parameter
    Eigen::VectorXd _upper_95_est;   ///< Upper 95% credibility interval for each parameter
    Eigen::VectorXd _lower_5_est;    ///< Lower 95% credibility interval for each parameter
    Eigen::MatrixXd _t_chain;        ///< Transformed chains for each parameter
    Eigen::VectorXd _t_mean_est;     ///< Transformed mean values for each parameter
    Eigen::VectorXd _t_median_est;   ///< Transformed median values for each parameter
    Eigen::VectorXd _t_upper_95_est; ///< Transformed upper 95% credibility interval for each parameter
    Eigen::VectorXd _t_lower_5_est;  ///< Transformed lower 95% credibility interval for each parameter
};

/**
 * @class BBVISSResults results.hpp
 */
class BBVISSResults : public Results {
public:
    /**
     * @brief Constructor for BBVISSResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ses
     * @param signal
     * @param scores
     * @param elbo_records
     * @param states
     * @param states_var
     */
    BBVISSResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                  const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                  std::vector<double> index, bool multivariate_model, double objective_value, std::string method,
                  bool z_hide, int max_lag, Eigen::VectorXd ses, Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
                  std::optional<Eigen::VectorXd> scores     = std::nullopt,
                  Eigen::VectorXd elbo_records              = Eigen::VectorXd::Zero(0),
                  std::optional<Eigen::VectorXd> states     = std::nullopt,
                  std::optional<Eigen::VectorXd> states_var = std::nullopt);

    /**
     * @brief Stream operator for BBVISSResults
     * @param stream Output stream
     * @param results BBVISSResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const BBVISSResults& results);

    /**
     * @brief Plots the ELBO progress (if present)
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_elbo(size_t width = 15, size_t height = 7) const;

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;

private:
    double _objective_value;
    Eigen::MatrixXd _ihessian;
    Eigen::VectorXd _ses;
    Eigen::VectorXd _elbo_records;
    Eigen::MatrixXd _chain;          ///< Chains for each parameter
    Eigen::VectorXd _mean_est;       ///< Mean values for each parameter
    Eigen::VectorXd _median_est;     ///< Median values for each parameter
    Eigen::VectorXd _upper_95_est;   ///< Upper 95% credibility interval for each parameter
    Eigen::VectorXd _lower_5_est;    ///< Lower 95% credibility interval for each parameter
    Eigen::MatrixXd _t_chain;        ///< Transformed chains for each parameter
    Eigen::VectorXd _t_mean_est;     ///< Transformed mean values for each parameter
    Eigen::VectorXd _t_median_est;   ///< Transformed median values for each parameter
    Eigen::VectorXd _t_upper_95_est; ///< Transformed upper 95% credibility interval for each parameter
    Eigen::VectorXd _t_lower_5_est;  ///< Transformed lower 95% credibility interval for each parameter
};

/**
 * @class LaplaceResults results.hpp
 */
class LaplaceResults : public Results {
public:
    /**
     * Constructor for LaplaceResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param signal
     * @param scores
     * @param states
     * @param states_var
     */
    LaplaceResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                   const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                   std::vector<double> index, bool multivariate_model,
                   std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                   int max_lag, Eigen::MatrixXd ihessian, Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
                   std::optional<Eigen::VectorXd> scores     = std::nullopt,
                   std::optional<Eigen::VectorXd> states     = std::nullopt,
                   std::optional<Eigen::VectorXd> states_var = std::nullopt);

    /**
     * @brief Stream operator for LaplaceResults
     * @param stream Output stream
     * @param results LaplaceResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const LaplaceResults& results);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;

private:
    Eigen::MatrixXd _chain;
    Eigen::MatrixXd _ihessian;
    Eigen::VectorXd _mean_est;       ///< Mean values for each parameter
    Eigen::VectorXd _median_est;     ///< Median values for each parameter
    Eigen::VectorXd _upper_95_est;   ///< Upper 95% credibility interval for each parameter
    Eigen::VectorXd _lower_5_est;    ///< Lower 95% credibility interval for each parameter
    Eigen::MatrixXd _t_chain;        ///< Transformed chains for each parameter
    Eigen::VectorXd _t_mean_est;     ///< Transformed mean values for each parameter
    Eigen::VectorXd _t_median_est;   ///< Transformed median values for each parameter
    Eigen::VectorXd _t_upper_95_est; ///< Transformed upper 95% credibility interval for each parameter
    Eigen::VectorXd _t_lower_5_est;  ///< Transformed lower 95% credibility interval for each parameter
};

/**
 * @class MCMCResults results.hpp
 */
class MCMCResults : public Results {
public:
    /**
     * @brief Constructor for MCMCResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param samples
     * @param mean_est
     * @param median_est
     * @param lower_95_est
     * @param upper_95_est
     * @param signal
     * @param scores
     * @param states
     * @param states_var
     */
    MCMCResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                const std::string& model_type, const LatentVariables& latent_variables, Eigen::MatrixXd data,
                std::vector<double> index, bool multivariate_model,
                std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
                Eigen::MatrixXd samples, Eigen::VectorXd mean_est, Eigen::VectorXd median_est,
                Eigen::VectorXd upper_95_est, Eigen::VectorXd lower_95_est,
                Eigen::VectorXd signal = Eigen::VectorXd::Zero(0), std::optional<Eigen::VectorXd> scores = std::nullopt,
                std::optional<Eigen::VectorXd> states     = std::nullopt,
                std::optional<Eigen::VectorXd> states_var = std::nullopt);

    /**
     * @brief Stream operator for MCMCResults
     * @param stream Output stream
     * @param results MCMCResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MCMCResults& results);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;

private:
    Eigen::MatrixXd _samples;
    Eigen::VectorXd _mean_est;
    Eigen::VectorXd _median_est;
    Eigen::VectorXd _lower_95_est;
    Eigen::VectorXd _upper_95_est;
};