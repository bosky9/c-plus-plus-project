#pragma once

#include "inference/norm_post_sim.hpp"
#include "latent_variables.hpp"
#include "matplotlibcpp.hpp"
#include "output/tableprinter.hpp"
#include "tests/nhst.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

class Results {
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
    // FIXME: _scores, _states e _states_var non sono mai usate con i modelli ARMA
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

public:
    virtual void summary(bool transformed) = 0;

    /**
     * @brief Returns latent variables
     * @return Latent variables in _z
     */
    [[nodiscard]] LatentVariables get_z() const;
};

// Public is necessary for pointers (Return* p = &MLEResults{...})
class MLEResults : public Results {
private:
    Eigen::VectorXd _results; // FIXME: OptimizeResult type in Python (da scipy) ma viene utilizzato solo l'array
                              // non gli altri oggetti al suo interno
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

public:
    /**
     * @brief Returns the Inverse Hessian matrix
     * @return Inverse Hessian matrix
     */
    [[nodiscard]] Eigen::MatrixXd get_ihessian() const;

public:
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
     * @param mleresults MLEResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MLEResults& mleresults);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};

class BBVIResults : public Results {
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
};

class BBVISSResults : public Results {
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
};

class LaplaceResults : public Results {
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
};

class MCMCResults : public Results {
private:
    Eigen::MatrixXd _samples;
    Eigen::VectorXd _mean_est;
    Eigen::VectorXd _median_est;
    Eigen::VectorXd _lower_95_est;
    Eigen::VectorXd _upper_95_est;

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
};