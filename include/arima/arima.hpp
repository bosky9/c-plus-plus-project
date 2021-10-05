#pragma once

#include "arima/arima_recursion.hpp"
#include "data_check.hpp"
#include "families/family.hpp"
#include "families/normal.hpp"
#include "headers.hpp"
#include "output/tableprinter.hpp"
#include "tests/nhst.hpp"
#include "tsm.hpp"

#include <map>
#include <type_traits>
#include <vector>

/**
 * @brief Mean function applied to a vector
 * @param v Vector of double
 * @return Mean of values inside the vector
 */
inline double mean(Eigen::VectorXd v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline std::vector<double> diff(const std::vector<double>& v) {
    std::vector<double> new_v(v.size() - 1);
    for (size_t i{0}; i < new_v.size(); i++)
        new_v.at(i) = v.at(i + 1) - v.at(i);
    return std::move(new_v);
}

inline double percentile(Eigen::VectorXd v, uint8_t p) {
    std::sort(v.begin(), v.end());
    double n{static_cast<double>((v.size() + 1) * p)};
    if (n == 1.0)
        return v[0];
    else if (n == static_cast<double>(v.size()))
        return v[v.size() - 1];
    else {
        Eigen::Index k{static_cast<Eigen::Index>(n)};
        return v[k - 1] + (n - static_cast<double>(k)) * (v[k] - v[k - 1]);
    }
}

template<typename Base, typename T>
inline bool instanceof (const T*) {
    return std::is_base_of_v<Base, T>;
}

/**
 * @brief AutoRegressive Integrated Moving Average (ARIMA) models
 * (inherits time series methods from the TSM parent class)
 */
class ARIMA : public TSM {
private:
    size_t _ar;     ///< How many AR lags the model will have
    size_t _ma;     ///< How many MA lags the model will have
    size_t _integ;  ///< How many times to difference the time series (default 0)
    size_t _target; ///< Which array index to use. By default, first array index will be selected as the dependent
                    ///< variable.
    std::unique_ptr<Family> _family; ///< E.g. Normal()
    Eigen::MatrixXd _x;
    std::function<double(double)> _link;
    bool _scale;
    bool _shape;
    bool _skewness;
    std::function<double(double)> _mean_transform; ///< A function which transforms the location parameter
    bool _cythonized;
    size_t _data_length;
    std::string _model_name2;
    size_t _family_z_no;

    /**
     * @brief Creates the Autoregressive Matrix for the model
     * @return Autoregressive Matrix
     */
    Eigen::MatrixXd ar_matrix();

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
    std::tuple<double, double, double> get_scale_and_shape(Eigen::VectorXd transformed_lvs) const;

    /**
     * @brief Obtains model scale, shape, skewness latent variables for a 2d array of simulations
     * @param transformed_lvs Transformed latent variable vector (2d - with draws of each variable)
     * @return Tuple of vectors (each being scale, shape and skewness draws)
     */
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    get_scale_and_shape_sim(Eigen::MatrixXd transformed_lvs);

    /**
     * @brief Creates the structure of the model (model matrices etc) for a Normal family ARIMA model
     * @param beta Contains untransformed starting values for the latent variables
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd> normal_model(Eigen::VectorXd beta);

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
    std::pair<Eigen::VectorXd, Eigen::VectorXd> non_normal_model(Eigen::VectorXd beta);

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
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mb_normal_model(Eigen::VectorXd beta, size_t mini_batch);

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
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mb_non_normal_model(Eigen::VectorXd beta, size_t mini_batch);

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
    //  TODO: Non abbiamo implementato in families Poisson!

    /**
     * @brief Calculates the negative log-likelihood of the model for Normal family
     * @param beta Contains untransformed starting values for latent variables
     * @return The negative logliklihood of the model
     */
    double normal_neg_loglik(Eigen::VectorXd beta);

    /**
     * @brief Calculates the negative log-likelihood of the model for Normal family for a minibatch
     * @param beta Contains untransformed starting values for latent variables
     * @param mini_batch Size of each mini batch of data
     * @return The negative logliklihood of the model
     */
    double normal_mb_neg_loglik(Eigen::VectorXd beta, size_t mini_batch);

    /**
     * @brief Calculates the negative log-likelihood of the model for non-Normal family
     * @param beta Contains untransformed starting values for latent variables
     * @return The negative logliklihood of the model
     */
    double non_normal_neg_loglik(Eigen::VectorXd beta);

    /**
     * @brief Calculates the negative log-likelihood of the model for non-Normal family for a minibatch
     * @param beta Contains untransformed starting values for latent variables
     * @param mini_batch Size of each mini batch of data
     * @return The negative logliklihood of the model
     */
    double non_normal_mb_neg_loglik(Eigen::VectorXd beta, size_t mini_batch);

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
    Eigen::VectorXd mean_prediction(Eigen::VectorXd mu, Eigen::VectorXd Y, size_t h, Eigen::VectorXd t_z);

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
    Eigen::MatrixXd sim_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                   Eigen::VectorXd t_params, size_t simulations);

    /**
     * @brief Simulates a h-step ahead mean prediction
     * @details Same as mean_prediction() but now we repeat the process  by a number of times (simulations) and shock
     * the process with random draws from the family, e.g. Normal shocks.
     * @param h How many steps ahead for the prediction
     * @param simulations How many simulations to perform
     * @return Matrix of simulations
     */
    Eigen::MatrixXd sim_prediction_bayes(long h, size_t simulations);

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
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
    summarize_simulations(Eigen::VectorXd mean_values, Eigen::VectorXd sim_vector, Eigen::VectorXd date_index, long h,
                          long past_values);

public:
    /**
     * @brief Constructor for ARIMA object
     * @param data The univariate time series data that will be used
     * @param index The times of the input data (years, days or seconds)
     * @param ar How many AR lags the model will have
     * @param ma How many MA lags the model will have
     * @param integ How many times to difference the time series (default 0)
     * @param family E.g. Normal() (default)
     */
    ARIMA(const std::vector<double>& data, const std::vector<double>& index, size_t ar, size_t ma, size_t integ = 0,
          Family* family = reinterpret_cast<Family*>(new Normal()));

    /**
     * @brief Constructor for ARIMA object
     * @param data The univariate time series data that will be used
     * @param target Which array index to use
     * @param index The times of the input data (years, days or seconds)
     * @param ar How many AR lags the model will have
     * @param ma How many MA lags the model will have
     * @param integ How many times to difference the time series (default 0)
     * @param family E.g. Normal() (default)
     */
    ARIMA(const std::map<std::string, std::vector<double>>& data, const std::vector<double>& index,
          const std::string& target, size_t ar, size_t ma, size_t integ = 0,
          Family* family = reinterpret_cast<Family*>(new Normal()));

    /**
     * @brief Creates the structure of the model (model matrices etc) for a general family ARIMA model
     * @param beta Contains untransformed starting values for the latent variables
     * @return Tuple of vectors:
     * - mu: contains the predicted values (location) for the time series
     * - Y: contains the length-adjusted time series (accounting for lags)
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd> model(const Eigen::VectorXd& beta);

    /**
     * @brief Plots the fit of the model against the data
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_fit(std::optional<size_t> width = 10, std::optional<size_t> height = 7);

    /**
     * @brief Plots forecasts with the estimated model
     * @param h How many steps ahead would you like to forecast
     * @param past_values How many past observations to show on the forecast graph
     * @param intervals Would you like to show prediction intervals for the forecast?
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_predict(size_t h = 5, size_t past_values = 20, bool intervals = true, std::optional<size_t> width = 10,
                      std::optional<size_t> height = 7);

    /**
     * @brief Makes dynamic out-of-sample predictions with the estimated model on in-sample data
     * @param h How many steps would you like to forecast
     * @param fit_once Fits only once before the in-sample prediction; if False, fits after every new datapoint
     * @param fit_method Which method to fit the model with
     * @param intervals Whether to return prediction intervals
     * @return Vector with predicted values
     */
    Eigen::VectorXd predict_is(size_t h = 5, bool fit_once = true, std::string fit_method = "MLE",
                               bool intervals = false);

    /**
     * @brief Plots forecasts with the estimated model against data
     * @param h How many steps would you like to forecast
     * @param fit_once Fits only once before the in-sample prediction; if False, fits after every new datapoint
     * @param fit_method Which method to fit the model with
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_predict_is(size_t h = 5, bool fit_once = true, std::string fit_method = "MLE",
                         std::optional<size_t> width = 10, std::optional<size_t> height = 7);

    /**
     * @brief Makes forecast with the estimated model
     * @param h How many steps would you like to forecast
     * @param intervals Whether to return prediction intervals
     * @return Vector with predicted values
     */
    Eigen::VectorXd predict(size_t h = 5, bool intervals = false);

    /**
     * @brief Samples from the posterior predictive distribution
     * @param nsims How many draws from the posterior predictive distribution
     * @return Array of draws from the data
     */
    Eigen::MatrixXd sample(size_t nsims = 1000);

    /**
     * @brief Plots draws from the posterior predictive density against the data
     * @param nsims How many draws from the posterior predictive distribution
     * @param plot_data Whether to plot the data or not
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_sample(size_t nsims = 10, bool plot_data = true, std::optional<size_t> width = 10,
                     std::optional<size_t> height = 7);

    /**
     * @brief Computes posterior predictive p-value
     * @param nsims How many draws for the PPC
     * @param T A discrepancy measure - e.g. mean, std or max
     * @return Posterior predictive p-value
     */
    double ppc(size_t nsims = 1000, std::function<double(Eigen::VectorXd)> T = mean);

    /**
     * @brief Plots histogram of the discrepancy from draws of the posterior
     * @param nsims How many draws for the PPC
     * @param T A discrepancy measure - e.g. mean, std or max
     * @param width Width of the figure
     * @param height Height of the figure
     */
    void plot_ppc(size_t nsims = 1000, std::function<double(Eigen::VectorXd)> T = mean,
                  std::optional<size_t> width = 10, std::optional<size_t> height = 7);
};