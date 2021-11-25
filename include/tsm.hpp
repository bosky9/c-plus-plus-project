#pragma once

#include "Eigen/Core"
#include "families/family.hpp"
#include "hessian.hpp"
#include "inference/bbvi.hpp"
#include "inference/metropolis_hastings.hpp"
#include "inference/stoch_optim.hpp"
#include "latent_variables.hpp"
#include "posterior.hpp"
#include "results.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Struct that represents the internal data of a time-series model
 */
struct SingleDataFrame final {
    std::vector<double> index; ///< The times of the input data (years, days or seconds)
    std::vector<double> data;  ///< The univariate time series data (values) that will be used
    std::string data_name;     ///< The names of the data
};

/**
 * @brief Struct that represents the model output
 *
 * @details This is used to translate the returning results of the
 *          python _categorize_model_output(self, z) function.
 *
 *          This structure will then be used inside each of the
 *          ...fit() methods.
 */
struct ModelOutput final {
    Eigen::VectorXd theta;
    Eigen::MatrixXd Y;
    std::optional<Eigen::VectorXd> scores;
    std::optional<Eigen::VectorXd> states;
    std::optional<Eigen::VectorXd> states_var;
    std::optional<std::vector<std::string>> X_names;
};

/**
 * @class TSM tsm.hpp
 * @brief Contains general time series methods to be inherited by models
 */
class TSM {
protected:
    SingleDataFrame _data_frame;
    std::string _model_name;
    std::string _model_name2; ///< The self.model_name2 variable in Python
    std::string _model_type;  ///< The type of model (e.g. 'ARIMA', 'GARCH')
    bool _multivariate_model;
    std::function<double(const Eigen::VectorXd&)>
            _neg_logposterior; /**<
                                *  This function is the equivalent of the neg_loposterior(self, beta) function in
                                * python.
                                *
                                *  This function is initialized with a lamba expression, which calls the
                                *  double neg_logposterior(Eigen::VectorXd beta) c++ function.
                                *
                                *  This function is necessary for the following methods:
                                *   - _bbvi_fit( ... ), used internally and passed as the first argument
                                *   - _mcmc_fit( ... ), used internally
                                *   - _optimize_fit( ... ), passed as the first argument
                                *   - _laplace_fit( ... ), passed as the first argument
                                *
                                *   This function utilizes the _neg_loglik function.
                                */

    std::function<double(const Eigen::VectorXd&, size_t)> _mb_neg_logposterior; ///< Similar to _neg_logposterior
    // std::function<double(Eigen::VectorXd)> _multivariate_neg_logposterior; // Only for VAR models

    std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const Eigen::VectorXd&)>
            _model; /**<
                     *  This function is initialized in the ARIMA class.
                     *  It is necessary for many of its functions.
                     *
                     *  It receives a vector as input; the size of this vector
                     *  must be of a specific size, related to the _data_frame field.
                     *  In fact, _model is only meant for internal usage.
                     */

    std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const Eigen::VectorXd&, size_t mb)>
            _mb_model; // Init in ARIMA
    std::function<double(const Eigen::VectorXd&)>
            _neg_loglik; /**<
                          *  This function is the equivalent of the python neg_loglik( ... ) function,
                          *  which is initialized in the ARIMA class.
                          *
                          *  This function is not initialized in the TSM constructor,
                          *  following the python implementation. It will be instead initialized in ARIMA.
                          *
                          *  This function is necessary for the following methods:
                          *   - _optimize_fit( ... ), passed as parameter and used internally
                          *   - neg_logposterior(beta), used internally
                          *
                          *  It would also be used in the multivariate_neg_logposterior(beta) method,
                          *  and the _ols_fit(...) one,
                          *  but we did not implement VAR models.
                          */

    std::function<double(const Eigen::VectorXd&, size_t mb)> _mb_neg_loglik; ///< Similar to _neg_loglik
    bool _z_hide;
    int _max_lag;
    LatentVariables _latent_variables; ///< Holding variables for model output
    size_t _z_no;
    double _norm_std;
    double _norm_mean;
    std::string _default_method;                 ///< Default method for fitting
    std::vector<std::string> _supported_methods; ///< Supported methods for fitting
    bool _use_ols_covariance;
    size_t _ylen;
    bool _is_pandas;

    explicit TSM(const std::string& model_type);

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
    [[nodiscard]] virtual ModelOutput categorize_model_output(const Eigen::VectorXd& z) const = 0;

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
    BBVIResults* _bbvi_fit(const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& posterior,
                           const std::string& optimizer = "RMSProp", size_t iterations = 1000, bool map_start = true,
                           size_t batch_size = 12, std::optional<size_t> mini_batch = 12, double learning_rate = 0.001,
                           bool record_elbo = false, bool quiet_progress = false,
                           const std::optional<Eigen::VectorXd>& start = std::nullopt);

    /**
     * @brief Performs a Laplace approximation to the posterior
     * @param obj_type method, whether a likelihood or a posterior
     * @return A LaplaceResults object
     */
    LaplaceResults* _laplace_fit(const std::function<double(Eigen::VectorXd)>& obj_type);

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
     *
     * @details The sampler.sample() returns a Sample structure
     *          which contains the equivalent of {chain, mean_est, median_est,
     *          upper_95_est, lower_95_est} python data.
     */
    MCMCResults* _mcmc_fit(double scale = 1.0, size_t nsims = 10000, const std::string& method = "M-H",
                           std::optional<Eigen::MatrixXd>& cov_matrix = (std::optional<Eigen::MatrixXd>&) std::nullopt,
                           bool map_start = true, bool quiet_progress = false);

    /**
     * @brief Performs OLS (not actually implemented)
     * @return A MLEResults object
     */
    virtual MLEResults* _ols_fit(); // Defined in VAR

    /**
     * @brief Fits models using Maximum Likelihood or Penalized Maximum Likelihood
     * @param obj_type method
     * @return A MLEResults object
     */
    MLEResults* _optimize_fit(const std::string& method, const std::function<double(Eigen::VectorXd)>& obj_type = {},
                              std::optional<bool> preopt_search           = true,
                              const std::optional<Eigen::VectorXd>& start = std::nullopt);

    /**
     * @brief Returns negative log posterior
     * @param beta Contains untransformed starting values for latent variables
     * @return Negative log posterior
     *
     * @details This is the function called by _neg_logposterior.
     *          It sums the logpdf of the prior of the latent_variables_plots.
     *          It also employs _neg_loglik, to initialize the sum.
     */
    [[nodiscard]] double neg_logposterior(const Eigen::VectorXd& beta);

    /**
     * @brief Returns negative log posterior
     * @param beta Contains untransformed starting values for latent variables
     * @param mini_batch Batch size for the data
     * @return Negative log posterior
     *
     * @details This is the function called by _mb_neg_logposterior.
     */
    [[nodiscard]] double mb_neg_logposterior(const Eigen::VectorXd& beta, size_t mini_batch);

    // Used only in VAR models
    //[[nodiscard]] double multivariate_neg_logposterior(const Eigen::VectorXd& beta);

public:
    //@TODO: consider using only optional on None parameters
    /**
     * @brief Fits a model
     * @param method A fitting method (e.g. 'MLE')
     * @return Results of the fit
     *
     * @detail  Since the Python function receives a list of kwargs,
     *          we decided to translate it explicitly as parameters,
     *          some of them tagged as "optional" because by default
     *          the Python version inits them as None.
     *
     *          This function calls all other _..._fit(...) functions.
     *
     *          Since the return type is "Results*" (Results being an abstract class),
     *          in order to return pointer to Results subclasses
     *          it is necessary to declare their extension as public.
     *          es. "class MLEResults : public Results {...}".
     */
    Results* fit(std::string method                         = "",
                 std::optional<Eigen::MatrixXd>& cov_matrix = (std::optional<Eigen::MatrixXd>&) std::nullopt,
                 std::optional<size_t> iterations = 1000, std::optional<size_t> nsims = 1000,
                 const std::optional<std::string>& optimizer = "RMSProp", std::optional<size_t> batch_size = 12,
                 std::optional<size_t> mini_batch = std::nullopt, std::optional<bool> map_start = true,
                 std::optional<double> learning_rate = 0.001, std::optional<bool> record_elbo = std::nullopt,
                 std::optional<bool> quiet_progress = false);

    /**
     * @brief Auxiliary function for creating dates for forecasts
     * @param n How many steps to forecast
     * @return A transformed date_index object
     */
    [[nodiscard]] std::vector<double> shift_dates(size_t n) const;

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
    void plot_z(const std::optional<std::vector<size_t>>& indices = std::nullopt, size_t width = 15,
                size_t height = 5) const;

    // Not used in Python
    // void plot_parameters(const std::optional<std::vector<size_t>>& indices = std::nullopt, size_t width = 15, size_t
    // height = 5);

    /**
     * @brief Adjusts priors for the latent variables
     * @param index Which latent variable index/indices to be altered
     * @param prior Which prior distribution? E.g. Normal(0,1)
     */
    void adjust_prior(const std::vector<size_t>& index, const Family& prior);

    /**
     * @brief Draws latent variables from the model (for Bayesian inference)
     * @param nsims How many draws to take
     * @return Matrix of draws
     */
    [[nodiscard]] Eigen::MatrixXd draw_latent_variables(size_t nsims = 5000) const;

    /**
     * @brief Returns the latent variables
     * @return Latent variables
     */
    [[nodiscard]] virtual LatentVariables get_latent_variables() const;
};