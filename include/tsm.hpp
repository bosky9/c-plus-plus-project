#pragma once

#include "headers.hpp"
#include "inference/bbvi.hpp"
#include "latent_variables.hpp"
#include "results.hpp"

#include <functional>
#include <optional>
#include <string>

/**
 * @brief Struct that represents the model output
 */
struct ModelOutput final {
    Eigen::VectorXd theta;
    Eigen::MatrixXd y;
    Eigen::VectorXd scores;
    Eigen::VectorXd states;
    Eigen::VectorXd states_var;
    std::vector<std::string> x_names;
};

/**
 * @brief Contains general time series methods to be inherited by models
 */
class TSM {
protected:
    std::vector<std::string> _data_name;
    std::string _model_name;
    std::string _model_type; ///< The type of model (e.g. 'ARIMA', 'GARCH')
    std::vector<size_t> _index;
    bool _multivariate_model;
    std::function<double(Eigen::VectorXd)> _neg_logposterior;
    std::function<double(Eigen::VectorXd)> _neg_loglik;
    // Not used in Python
    // std::function<double(Eigen::VectorXd)> _multivariate_neg_logposterior;
    std::function<double(Eigen::VectorXd)> _mb_neg_logposterior; // TODO: Check function parameters
    std::function<double(Eigen::VectorXd)> _mb_neg_loglik;       // TODO: Check function parameters
    bool _z_hide;
    int _max_lag;
    LatentVariables _latent_variables; ///< Holding variables for model output
    size_t _z_no;
    Eigen::MatrixXd _data;
    double _norm_std;
    double _norm_mean;
    std::string _default_method;                 ///< Default method for fitting
    std::vector<std::string> _supported_methods; ///< Supported methods for fitting
    bool _use_ols_covariance;
    size_t _ylen;
    bool _is_pandas;

    TSM(const std::string& model_type);

    // TODO: I seguenti metodi sono presenti solo nella sottoclasse VAR
    //  Limitare i metodi che li usano solo alla classe VAR ?
    //_create_B_direct();
    //_ols_covariance();
    //_estimator_cov();
    //_preoptimize_model();
    //_custom_covariance();

    // TODO: Implement this function for each subclass of TSM (each model return different data)
    /**
     * @brief Return output data of the model
     * @param z Untransformed starting values for the latent variables
     * @return A ModelOutput object
     */
    [[nodiscard]] virtual ModelOutput _categorize_model_output(const Eigen::VectorXd& z) const = 0;

    /**
     * @brief Performs Black Box Variational Inference
     * @param posterior Hands _bbvi_fit a posterior object
     * @param optimizer Stochastic optimizer: one of RMSProp or ADAM
     * @param iterations How many iterations for BBVI
     * @param map_start Whether to start values from a MAP estimate (if False, uses default starting values)
     * @param batch_size
     * @param mini_batch
     * @param learning_rate
     * @param record_elbo
     * @param quiet
     * @return A BBVIResults object
     */
    BBVIResults _bbvi_fit(const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& posterior,
                          const std::string& optimizer = "RMSProp", size_t iterations = 1000, bool map_start = true,
                          size_t batch_size = 12, std::optional<size_t> mini_batch = std::nullopt,
                          double learning_rate = 0.001, bool record_elbo = false, bool quiet_progress = false,
                          const Eigen::VectorXd& start = Eigen::VectorXd::Zero(0));

    /**
     * @brief Performs a Laplace approximation to the posterior
     * @param obj_type method, whether a likelihood or a posterior
     * @return A LaplaceResults object
     */
    LaplaceResults _laplace_fit(const std::function<double(Eigen::VectorXd)>& obj_type);

    /**
     * @brief Performs random walk Metropolis-Hastings
     * @param scale Default starting scale
     * @param nsims Number of simulations
     * @param printer Whether to print results or not
     * @param method What type of MCMC
     * @param cov_matrix Can optionally provide a covariance matrix for M-H
     * @param map_start
     * @param quiet_progress
     * @return A MCMCResults object
     */
    MCMCResults _mcmc_fit(double scale = 1.0, size_t nsims = 10000, bool printer = true,
                          const std::string& method                        = "M-H",
                          const std::optional<Eigen::MatrixXd>& cov_matrix = std::nullopt, bool map_start = true,
                          bool quiet_progress = false);

    /**
     * @brief Performs OLS
     * @return A MLEResults object
     */
    MLEResults _ols_fit();

    /**
     * @brief Fits models using Maximum Likelihood or Penalized Maximum Likelihood
     * @param obj_type method
     * @return A MLEResults object
     */
    MLEResults _optimize_fit(const std::function<double(Eigen::VectorXd)>& obj_type = {});

public:
    /**
     * @brief Fits a model
     * @param method A fitting method (e.g. 'MLE')
     * @return Results of the fit
     */
    Results* fit(const std::string& method = "");

    /**
     * @brief Auxiliary function for creating dates for forecasts
     * @param h How many steps to forecast
     * @return A transformed date_index object
     */
    std::vector<size_t> shift_dates(size_t h);

    /**
     * @brief Transforms latent variables to actual scale by applying link function
     * @return Transformed latent variables
     */
    [[nodiscard]] Eigen::VectorXd transform_z() const;

    // Not used in Python
    // [[nodiscard]] Eigen::VectorXd transform_parameters() const;

    /**
     * @brief Plots latent variables by calling latent parameters object
     * @param indices Vector of indices to plot
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_z(const std::optional<std::vector<size_t>>& indices = std::nullopt, size_t width = 15, size_t height = 5);

    // Not used in Python
    // void plot_parameters(const std::optional<std::vector<size_t>>& indices = std::nullopt, size_t width = 15, size_t
    // height = 5);

    /**
     * @brief Adjusts priors for the latent variables
     * @param index Which latent variable index/indices to be altered
     * @param prior Which prior distribution? E.g. Normal(0,1)
     */
    void adjust_prior(const std::vector<size_t>& index, Family& prior);

    /**
     * @brief Draws latent variables from the model (for Bayesian inference)
     * @param nsims How many draws to take
     * @return Matrix of draws
     */
    Eigen::MatrixXd draw_latent_variables(size_t nsims = 5000);
};