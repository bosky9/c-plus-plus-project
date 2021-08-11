#pragma once

#include <random>
#include <chrono>
#include <cassert>
#include <memory>

#include "../headers.hpp"
#include "family.hpp"

/**
 * @brief Normal distribution for time series
 */
class Normal : Family {
private:
    double mu0;
    double sigma0;
    std::string transform;
    short int param_no;
    bool covariance_prior;
    // gradient_only won't be used (no GAS models)

public:

    //@TODO: add Flat return value
    // Necessary for "build_latent_variables()" function
    struct lv_to_build {
        std::string name = "Normal scale";
        //Flat
        std::unique_ptr<Normal> n {new Normal(0.0, 3.0)};
        double zero = 0;
    };

    /**
     * @brief Constructor for Normal distribution
     * @param mu Mean for the Normal distribution
     * @param sigma Standard deviation for the Normal distribution
     * @param transform Whether to apply a transformation for the location latent variable
     *  (e.g. "exp" or "logit")
     */
    Normal(double mu = 0.0, double sigma = 1.0, std::string transform = "");

    /**
     * @brief Creates approximating Gaussian state space model for the Normal measurement density
     * @param beta Not actually used
     * @param T Not actually used
     * @param Z Not actually used
     * @param R Not actually used
     * @param Q Not actually used
     * @param h_approx Variance of the measurement density
     * @param data Univariate time series data (define the size of the created matrices)
     * * @return An array consisting of two Eigen::MatrixXd:
     *  - H: approximating measurement variance matrix
     *  - mu: approximating measurement constants
     */
    Eigen::MatrixXd *approximating_model(const std::vector<double> &beta, const Eigen::MatrixXd &T,
                                         const Eigen::MatrixXd &Z, const Eigen::MatrixXd &R, const Eigen::MatrixXd &Q,
                                         double h_approx, const std::vector<double> &data);

    /**
     * @brief Creates approximating Gaussian state space model for the Normal measurement density.
     * @param beta Not actually used
     * @param T Not actually used
     * @param Z Not actually used
     * @param R Not actually used
     * @param Q Not actually used
     * @param h_approx Variance of the measurement density
     * @param data Univariate time series data (define the size of the created matrices)
     * @param X Not actually used
     * @param state_no Not actually used
     * @return An array consisting of two Eigen::MatrixXd:
     *  - H: approximating measurement variance matrix
     *  - mu: approximating measurement constants
     */
    Eigen::MatrixXd *approximating_model_reg(const std::vector<double> &beta, const Eigen::MatrixXd &T,
                                             const Eigen::MatrixXd &Z, const Eigen::MatrixXd &R,
                                             const Eigen::MatrixXd &Q,
                                             double h_approx, const std::vector<double> &data,
                                             const std::vector<double> &X, int state_no);

    /**
     * @brief Builds additional latent variables for this family in a probabilistic model
     * @return A list of structs (each struct contains latent variable information)
     */
    static std::list<lv_to_build> build_latent_variables();

    /**
     * @brief Draws random variables from this distribution with new latent variables
     * @param loc Location parameter for the distribution
     * @param scale Scale parameter for the distribution
     * @param shape Not actually used
     * @param skewness Not actually used
     * @param nsims Number of draws to take from the distribution
     * @return Random draws from the distribution
     */
    std::vector<double> draw_variable(double loc, double scale, double shape,
                                      double skewness, int nsims);

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param size How many simulations to perform
     * @return Array of Normal random variable
     */
    std::vector<double> draw_variable_local(int size);

    /**
     * @brief Log PDF for Normal prior
     * @param mu Latent variable for which the prior is being formed over
     * @return log(p(mu))
     */
    double logpdf(double mu);

    /**
     * @brief Markov blanket for each likelihood term - used for space state models
     * @param y Univariate time series
     * @param mean Array of location parameters for the Normal distribution
     * @param scale Scale parameter for the Normal distribution
     * @param shape Tail thickness parameter for Normal distribution
     * @param skewness Skewness parameter for the Normal distribution
     * @return Markov blanket of the Normal family
     */
    static std::vector<double>
    markov_blanket(const std::vector<double> &y, const std::vector<double> &mean, double scale,
                   double shape, double skewness);

    /**
     * @brief Returns the attributes of this family if using in a probabilistic model
     * @return A struct with attributes of the family
     */
    static FamilyAttributes setup();

    /**
     * @brief Negative loglikelihood function for this distribution
     * @param y Univariate time series
     * @param mean Array of location parameters for the Normal distribution
     * @param scale Scale parameter for the Normal distribution
     * @param shape Tail thickness parameter for Normal distribution
     * @param skewness Skewness parameter for the Normal distribution
     * @return Negative loglikelihood of the Normal family
     */
    static double neg_loglikelihood(const std::vector<double> &y, const std::vector<double> &mean, double scale,
                                    double shape, double skewness);

    /**
     * @brief PDF for Normal prior
     * @param mu Latent variable for which the prior is being formed over
     * @return p(mu)
     */
    double pdf(double mu);

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @param value What to change the latent variable to
     */
    void vi_change_param(int index, double value);

    /**
     * @brief Wrapper function for selecting appropriate latent variable for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @return The appropriate indexed parameter
     */
    double vi_return_param(int index);

    /**
     * @brief Return the gradient of the location latent variable mu
     *  (used for variational inference)
     * @param value A random variable
     * @return The gradient of the location latent variable mu at x
     */
    double vi_loc_score(double x);

    /**
     * @brief Return the score of the scale, where scale = exp(x)
     *  (used for variational inference)
     * @param value A random variable
     * @return The gradient of the scale latent variable at x
     */
    double vi_scale_score(double x);

    /**
     * @brief Wrapper function for selecting appropriate score
     * @param value A random variable
     * @param index 0 or 1 depending on which latent variable
     * @return The gradient of the scale latent variable at x
     */
    double vi_score(double x, int index);
};