#pragma once

#include "arima/arima_recursion.hpp"
#include "data_check.hpp"
#include "families/family.hpp"
#include "families/normal.hpp"
#include "headers.hpp"
#include "matplotlibcpp.hpp"
#include "multivariate_normal.hpp"
#include "output/tableprinter.hpp"
#include "tests/nhst.hpp"
#include "tsm.hpp"
#include "utilities.hpp"

#include <map>
#include <type_traits>
#include <vector>

namespace plt = matplotlibcpp;

/**
 * @brief Mean function applied to a vector
 * @param v Vector of double
 * @return Mean of values inside the vector
 */
inline double mean(Eigen::VectorXd v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline double median(Eigen::VectorXd v) {
    Eigen::Index n{v.size() / 2};
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

inline double max(Eigen::VectorXd v) {
    return *std::max_element(v.begin(), v.end());
}


inline double min(Eigen::VectorXd v) {
    return *std::min_element(v.begin(), v.end());
}

inline std::vector<double> diff(const std::vector<double>& v) {
    std::vector<double> new_v(v.size() - 1);
    for (size_t i{0}; i < new_v.size(); i++)
        new_v.at(i) = v.at(i + 1) - v.at(i);
    return std::move(new_v);
}

inline double percentile(Eigen::VectorXd v, uint8_t p) {
    std::sort(v.begin(), v.end());
    size_t index{static_cast<size_t>(std::ceil(p * 0.01 * v.size()))};
    return v(index - 1);
}

/**
 * @class ARIMA arima.hpp
 * @brief AutoRegressive Integrated Moving Average (ARIMA) models
 * (inherits time series methods from the TSM parent class)
 */
class ARIMA : public TSM {
private:
    size_t _ar;    ///< How many AR lags the model will have
    size_t _ma;    ///< How many MA lags the model will have
    size_t _integ; ///< How many times to difference the time series (default 0)
    std::vector<double> _data_original;
    size_t _data_length;
    Eigen::MatrixXd _x;
    std::unique_ptr<Family> _family; ///< E.g. Normal()
    std::function<double(double)> _link;
    bool _scale;
    bool _shape;
    bool _skewness;
    std::function<double(double)> _mean_transform; ///< A function which transforms the location parameter
    std::string _model_name2;
    size_t _family_z_no;

    /**
     * @brief Creates the Autoregressive Matrix for the model
     * @return Autoregressive Matrix
     */
    Eigen::MatrixXd ar_matrix();

    /**
     * @brief Return output data of the model
     * @param z Untransformed starting values for the latent variables
     * @return A ModelOutput object
     */
    [[nodiscard]] ModelOutput categorize_model_output(const Eigen::VectorXd& z) const override;

    /**
     * @brief Creates the model's latent variables
     * @details A latent variable addition requires:
        - Latent variable name - e.g. Constant
        - Latent variable prior family - e.g. Normal(0, 1)
        - Variational approximation - e.g. Normal(0, 1)
     */
    void create_latent_variables();

    /**
     * @brief Obtains model scale, shape and skewness latent variables
     * @param transformed_lvs Transformed latent variable vector
     * @return Tuple of model scale, model shape, model skewness
     */
    [[nodiscard]] std::tuple<double, double, double> get_scale_and_shape(const Eigen::VectorXd& transformed_lvs) const;

    /**
     * @brief Obtains model scale, shape, skewness latent variables for a 2d array of simulations
     * @param transformed_lvs Transformed latent variable vector (2d - with draws of each variable)
     * @return Tuple of vectors (each being scale, shape and skewness draws)
     */
    [[nodiscard]] std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    get_scale_and_shape_sim(const Eigen::MatrixXd& transformed_lvs) const;

    /**
     * @brief Creates the structure of the model (model matrices etc) for a Normal family ARIMA model
     * @param beta Contains untransformed starting values for the latent variables
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> normal_model(const Eigen::VectorXd& beta) const;

    /**
     * @brief Creates the structure of the model (model matrices etc) for a Poisson model.
     * @details Here we apply a link function to the MA lags.
     * @param beta Contains untransformed starting values for the latent variables
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    // std::pair<Eigen::VectorXd, Eigen::VectorXd> poisson_model(Eigen::VectorXd beta);
    //  TODO: Non abbiamo implementato in families Poisson!

    /**
     * @brief Creates the structure of the model (model matrices etc) for a non-normal model.
     * Here we apply a link function to the MA lags.
     * @param beta Contains untransformed starting values for the latent variables
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> non_normal_model(const Eigen::VectorXd& beta) const;

    /**
     * @brief Creates the structure of the model (model matrices etc) for mini batch model.
     * @details Here the structure is the same as for normal_model() but we are going to sample a random choice of data
     * points (of length mini_batch).
     * @param beta Contains untransformed starting values for the latent variables
     * @param mini_batch Mini batch size for the data sampling
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> mb_normal_model(const Eigen::VectorXd& beta,
                                                                              size_t mini_batch) const;

    /**
     * @brief Creates the structure of the model (model matrices etc) for mini batch model.
     * @details Here the structure is the same as for non_normal_model() but we are going to sample a random choice of
     * data points (of length mini_batch).
     * @param beta Contains untransformed starting values for the latent variables
     * @param mini_batch Mini batch size for the data sampling
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::VectorXd> mb_non_normal_model(const Eigen::VectorXd& beta,
                                                                                  size_t mini_batch) const;

    /**
     * @brief Creates the structure of the model (model matrices etc) for mini batch model.
     * @details Here the structure is the same as for poisson_model() but we are going to sample a random choice of
     * data points (of length mini_batch).
     * @param beta Contains untransformed starting values for the latent variables
     * @param mini_batch Mini batch size for the data sampling
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    // std::pair<Eigen::VectorXd, Eigen::VectorXd> mb_poisson_model(Eigen::VectorXd beta, size_t mini_batch);
    // No Poisson implementation.

    /**
     * @brief Calculates the negative log-likelihood of the model for Normal family for a minibatch
     * @param beta Contains untransformed starting values for latent variables
     * @param mini_batch Size of each mini batch of data
     * @return The negative logliklihood of the model
     */
    [[nodiscard]] double normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const;


    /**
     * @brief Calculates the negative log-likelihood of the model for non-Normal family for a minibatch
     * @param beta Contains untransformed starting values for latent variables
     * @param mini_batch Size of each mini batch of data
     * @return The negative logliklihood of the model
     */
    [[nodiscard]] double non_normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const;

    /**
     * @brief Creates a h-step ahead mean prediction
     * @details This function is used for predict(). We have to iterate over the number of timepoints (h) that the user
     * wants to predict, using as inputs the ARIMA parameters, past datapoints, and past predicted datapoints.
     * @param mu The past predicted values
     * @param Y The past data
     * @param h How many steps ahead for the prediction
     * @param t_z A vector of (transformed) latent variables
     * @return h-length vector of mean predictions
     */
    [[nodiscard]] Eigen::VectorXd mean_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                                  Eigen::VectorXd t_z) const;

    /**
     * @brief Simulates a h-step ahead mean prediction
     * @details Same as mean_prediction() but now we repeat the process  by a number of times (simulations) and shock
     * the process with random draws from the family, e.g. Normal shocks.
     * @param mu The past predicted values
     * @param Y The past data
     * @param h How many steps ahead for the prediction
     * @param t_params A vector of (transformed) latent variables
     * @param simulations How many simulations to perform
     * @return Matrix of simulations
     */
    [[nodiscard]] Eigen::MatrixXd sim_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                                 const Eigen::VectorXd& t_params, size_t simulations) const;

    /**
     * @brief Simulates a h-step ahead mean prediction
     * @details Same as mean_prediction() but now we repeat the process  by a number of times (simulations) and shock
     * the process with random draws from the family, e.g. Normal shocks.
     * @param h How many steps ahead for the prediction
     * @param simulations How many simulations to perform
     * @return Matrix of simulations
     */
    [[nodiscard]] Eigen::MatrixXd sim_prediction_bayes(size_t h, size_t simulations) const;

    /**
     * @brief Produces simulation forecasted values and prediction intervals
     * @details This is a utility function that constructs the prediction intervals and other quantities used for
     * plot_predict() in particular.
     * @param mean_values Mean predictions for h-step ahead forecasts
     * @param sim_vector N simulated predictions for h-step ahead forecasts
     * @param date_index Date index for the simulations
     * @param h How many steps ahead to forecast
     * @param past_values How many past observations to include in the forecast plot
     * @return Tuple of vectors: error bars, forecasted values, values and indices to plot
     */
    [[nodiscard]] std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
    summarize_simulations(const Eigen::VectorXd& mean_values, const Eigen::MatrixXd& sim_vector,
                          const std::vector<double>& date_index, size_t h, size_t past_values) const;

public:
    /**
     * @brief Constructor for ARIMA object
     * @param data The univariate time series data that will be used
     * @param index The times of the input data (years, days or seconds)
     * @param ar How many AR lags the model will have
     * @param ma How many MA lags the model will have
     * @param integ How many times to difference the time series (default 0)
     * @param family E.g. Normal() (default). It must be moved from the caller.
     *
     * @details Notes:
     *
     *              The info about the data is kept inside a SingleDataFrame struct,
     *              called _data_frame, which yields the "data", "data_name", "index"
     *              python variables.
     *
     *              In order to assign a subclass to the unique pointer to Family (which is abstract),
     *              the method clone creates a deep copy of the passed family; the unique ptr is then
     *              reset to the new one.
     *              Please notice that the passed family is a unique_ptr, as to avoid possible memory leaks.
     *              This means that it must be moved from the caller, by using std::move().
     *              The caller then loses the ownership of the pointer.
     */
    ARIMA(const std::vector<double>& data, size_t ar, size_t ma, size_t integ = 0,
          std::unique_ptr<Family> family = std::make_unique<Normal>(0, 1));

    /**
     * @brief Calculates the negative log-likelihood of the model for non-Normal family
     * @param beta Contains untransformed starting values for latent variables
     * @return The negative logliklihood of the model
     */
    [[nodiscard]] double non_normal_neg_loglik(const Eigen::VectorXd& beta) const;

    [[nodiscard]] double normal_neg_loglik(const Eigen::VectorXd& beta) const;

    /**
     * @brief Constructor for ARIMA object
     * @param data_frame The input data for the model
     * @param ar How many AR lags the model will have
     * @param ma How many MA lags the model will have
     * @param integ How many times to difference the time series (default 0)
     * @param family E.g. Normal() (default)
     * @param target Which array index to use
     */
    ARIMA(const DataFrame& data_frame, size_t ar, size_t ma, size_t integ = 0,
          std::unique_ptr<Family> family = std::make_unique<Normal>(0,1),
          const std::string& target = "");

    /**
     * @brief Plots the fit of the model against the data
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_fit(std::optional<size_t> width = 10, std::optional<size_t> height = 7) const;

    /**
     * @brief Plots forecasts with the estimated model
     * @param h How many steps ahead would you like to forecast
     * @param past_values How many past observations to show on the forecast graph
     * @param intervals Would you like to show prediction intervals for the forecast?
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_predict(size_t h = 5, size_t past_values = 20, bool intervals = true, std::optional<size_t> width = 10,
                      std::optional<size_t> height = 7) const;

    /**
     * @brief Makes dynamic out-of-sample predictions with the estimated model on in-sample data
     * @param h How many steps would you like to forecast
     * @param fit_once Fits only once before the in-sample prediction; if False, fits after every new datapoint
     * @param fit_method Which method to fit the model with
     * @param intervals Whether to return prediction intervals
     * @return Vector with predicted values
     */
    [[nodiscard]] DataFrame predict_is(size_t h = 5, bool fit_once = true, const std::string& fit_method = "MLE",
                                       bool intervals = false) const;

    /**
     * @brief Plots forecasts with the estimated model against data
     * @param h How many steps would you like to forecast
     * @param fit_once Fits only once before the in-sample prediction; if False, fits after every new datapoint
     * @param fit_method Which method to fit the model with
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_predict_is(size_t h = 5, bool fit_once = true, const std::string& fit_method = "MLE",
                         std::optional<size_t> width = 10, std::optional<size_t> height = 7) const;

    /**
     * @brief Makes forecast with the estimated model
     * @param h How many steps would you like to forecast
     * @param intervals Whether to return prediction intervals
     * @return Vector with predicted values
     */
    [[nodiscard]] DataFrame predict(size_t h = 5, bool intervals = false) const;

    /**
     * @brief Samples from the posterior predictive distribution
     * @param nsims How many draws from the posterior predictive distribution
     * @return Array of draws from the data
     */
    [[nodiscard]] Eigen::MatrixXd sample(size_t nsims = 1000) const;

    /**
     * @brief Plots draws from the posterior predictive density against the data
     * @param nsims How many draws from the posterior predictive distribution
     * @param plot_data Whether to plot the data or not
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_sample(size_t nsims = 10, bool plot_data = true, std::optional<size_t> width = 10,
                     std::optional<size_t> height = 7) const;

    /**
     * @brief Computes posterior predictive p-value
     * @param nsims How many draws for the PPC
     * @param T A discrepancy measure - e.g. mean, std or max
     * @return Posterior predictive p-value
     */
    double ppc(size_t nsims = 1000, const std::function<double(Eigen::VectorXd)>& T = mean) const;

    /**
     * @brief Plots histogram of the discrepancy from draws of the posterior
     * @param nsims How many draws for the PPC
     * @param T A discrepancy measure - e.g. mean, std or max
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_ppc(size_t nsims = 1000, const std::function<double(Eigen::VectorXd)>& T = mean,
                  const std::string& T_name = "mean", std::optional<size_t> width = 10,
                  std::optional<size_t> height = 7) const;
};