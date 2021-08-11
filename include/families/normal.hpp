#pragma once

#include "../headers.hpp"
#include "family.hpp"
#include "family_attributes.hpp"

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

    // Necessary for "build_latent_variables()" function
    struct lv_to_build {
        std::string name;
        //Flat
        Normal* n = new Normal{0, 3};
        double zero = 0;
        ~lv_to_build() {delete n;}
    };

    /**
     * @brief Constructor for Normal distribution
     * @param mu (double): mean for the Normal distribution
     * @param sigma (double): standard deviation for the Normal distribution
     * @param transform (string): whether to apply a transformation for the location latent variable
     *  (e.g. "exp" or "logit")
     */
    Normal(double mu = 0.0, double sigma = 1.0, std::string transform = "");

    /**
     * @brief Creates approximating Gaussian state space model for the Normal measurement density.
     * @param beta Not actually used
     * @param T Not actually used
     * @param Z Not actually used
     * @param R Not actually used
     * @param Q Not actually used
     * @param h_approx Not actually used
     * @param data Define the size of the created matrices
     * @return H_mu, an array consisting of two Eigen::MatrixXd
     */
    Eigen::MatrixXd* approximating_model(std::vector<double> beta, Eigen::MatrixXd T, Eigen::MatrixXd Z,
                                         Eigen::MatrixXd R, Eigen::MatrixXd Q, double h_approx, std::vector<double>  data);

    /**
     * @brief Creates approximating Gaussian state space model for the Normal measurement density.
     * @param beta Not actually used
     * @param T Not actually used
     * @param Z Not actually used
     * @param R Not actually used
     * @param Q Not actually used
     * @param h_approx Not actually used
     * @param data Define the size of the created matrices
     * @param X Not actually used
     * @param state_no Not actually used
     * @return An array consisting of two Eigen::MatrixXd
     */
    Eigen::MatrixXd* approximating_model_reg(std::vector<double> beta, Eigen::MatrixXd T, Eigen::MatrixXd Z,
                                         Eigen::MatrixXd R, Eigen::MatrixXd Q, double h_approx,
                                         std::vector<double> data, std::vector<double> X, int state_no);

    // "build_latent_variables()" returns a list of structs
    /**
     * @brief Builds additional latent variables for this family in a probabilistic model
     * @return A list of structs (each struct contains latent variable information)
     */
    std::list<lv_to_build> build_latent_variables();

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
     * @param size (int): How many simulations to perform
     * @return Array of Normal random variable
     */
    std::vector<double> draw_variable_local(int size);

    /**
     * @brief Log PDF for Normal prior
     * @param mu (double): Latent variable for which the prior is being formed over
     * @return log(p(mu))
     */
    double logpdf(double mu);

    /**
     * @brief Markov blanket for each likelihood term - used for space state models
     * @param y (vector<double>): univariate time series
     * @param mean (vector<double>): array of location parameters for the Normal distribution
     * @param scale (double): scale parameter for the Normal distribution
     * @param shape (double): tail thickness parameter for Normal distribution
     * @param skewness (double): skewness parameter for the Normal distribution
     * @return Markov blanket of the Normal family
     */
    static std::vector<double> markov_blanket(std::vector<double> y, std::vector<double> mean, double scale, double shape, double skewness);

    /**
     * @brief Returns the attributes of this family if using in a probabilistic model
     * @return A struct with attributes of the family
     */
    static FamilyAttributes setup();

    /**
     * @brief Negative loglikelihood function for this distribution
     * @param y (vector<double>): univariate time series
     * @param mean (vector<double>): array of location parameters for the Normal distribution
     * @param scale (double): scale parameter for the Normal distribution
     * @param shape (double): tail thickness parameter for Normal distribution
     * @param skewness (double): skewness parameter for the Normal distribution
     * @return Negative loglikelihood of the Normal family
     */
    static double neg_loglikelihood(std::vector<double> y, std::vector<double> mean, double scale, double shape, double skewness);

    /**
     * @brief PDF for Normal prior
     * @param mu (double): Latent variable for which the prior is being formed over
     * @return p(mu)
     */
    double pdf(double mu);

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param index (int): 0 or 1 depending on which latent variable
     * @param value (double): what to change the latent variable to
     */
    void vi_change_param(int index, double value);

    /**
     * @brief Wrapper function for selecting appropriate latent variable for variational inference
     * @param index (int): 0 or 1 depending on which latent variable
     * @return (double) the appropriate indexed parameter
     */
    double vi_return_param(int index);

    /**
     * @brief Return the gradient of the location latent variable mu
     *  (used for variational inference)
     * @param value (double) a random variable
     * @return (double) the gradient of the location latent variable mu at x
     */
    double vi_loc_score(double x);

    /**
     * @brief Return the score of the scale, where scale = exp(x)
     *  (used for variational inference)
     * @param value (double) a random variable
     * @return (double) the gradient of the scale latent variable at x
     */
    double vi_scale_score(double x);

    /**
     * @brief Wrapper function for selecting appropriate score
     * @param value (double) a random variable
     * @param index (int): 0 or 1 depending on which latent variable
     * @return (double) the gradient of the scale latent variable at x
     */
    double vi_score(double x, int index);
};