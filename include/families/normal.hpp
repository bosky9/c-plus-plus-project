#pragma once

#include <cassert>
#include <chrono>
#include <memory>
#include <random>

#include "../headers.hpp"
#include "family.hpp"
#include "flat.hpp"

/**
 * @brief Normal distribution for time series
 *
 * @details Since this project won't cover GAS models,
 *          the involved chunk of code has not been translated.
 */
class Normal final : public Family {
private:
    double _mu0; ///< The mean of the Gaussian
    double _sigma0; ///< The variance of the Gaussian
    short unsigned int _param_no;
    bool _covariance_prior;
    // gradient_only won't be used (no GAS models)

public:

    struct lv_to_build {
        std::string name = "Normal scale";
        Flat flat{"exp"};
        std::unique_ptr<Normal> n{new Normal(0.0, 3.0)};
        double zero = 0;
    };     ///<  Necessary for "build_latent_variables()" function

    /**
     * @brief Constructor for Normal distribution
     * @param mu Mean for the Normal distribution
     * @param sigma Standard deviation for the Normal distribution
     * @param transform Whether to apply a transformation for the location latent variable
     *  (e.g. "exp" or "logit")
     */
    explicit Normal(double mu = 0.0, double sigma = 1.0, const std::string& transform = "");

    /**
     * @brief Copy constructor for Normal distribution
     * @param normal A Normal object
     */
    Normal(const Normal& normal);

    /**
     * @brief Move constructor for Normal distribution
     * @param normal A Normal object
     */
    Normal(Normal&& normal) noexcept;

    /**
     * @brief Assignment operator for Normal distribution
     * @param normal A Normal object
     */
    Normal& operator=(const Normal& normal);

    /**
     * @brief Move assignment operator for Normal distribution
     * @param normal A Normal object
     */
    Normal& operator=(Normal&& normal) noexcept;

    /**
     * @brief Equal operator for Normal
     * @param normal1 First object
     * @param normal2 Second object
     * @return If the two objects are equal
     */
    friend bool operator==(const Normal& normal1, const Normal& normal2);

    /**
     * @brief Creates approximating Gaussian state space model for the Normal measurement density
     * @param beta Not actually used
     * @param T Not actually used
     * @param Z Not actually used
     * @param R Not actually used
     * @param Q Not actually used
     * @param h_approx Variance of the measurement density
     * @param data Univariate time series data (define the size of the created matrices)
     * * @return A pointer to an array consisting of two Eigen::MatrixXd:
     *  - H: approximating measurement variance matrix
     *  - mu: approximating measurement constants
     */
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    approximating_model(const Eigen::VectorXd& beta, const Eigen::MatrixXd& T, const Eigen::MatrixXd& Z,
                        const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q, double h_approx,
                        const Eigen::VectorXd& data);

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
     * @return A pointer to an array consisting of two Eigen::MatrixXd:
     *  - H: approximating measurement variance matrix
     *  - mu: approximating measurement constants
     */
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    approximating_model_reg(const Eigen::VectorXd& beta, const Eigen::MatrixXd& T, const Eigen::MatrixXd& Z,
                            const Eigen::MatrixXd& R, const Eigen::MatrixXd& Q, double h_approx,
                            const Eigen::VectorXd& data, const Eigen::VectorXd& X, int state_no);

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
    static Eigen::VectorXd draw_variable(double loc, double scale, double shape, double skewness, int nsims);

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param size How many simulations to perform
     * @return Array of Normal random variable
     */
    [[nodiscard]] Eigen::VectorXd draw_variable_local(size_t size) const override;

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
     *
     * @details The original python code is
     *
     *          return ss.norm.logpdf(y, loc=mean, scale=scale)
     *
     *          Which returns the log of the probability density function at y of the given RV.
     *          Since mean can be a vector (in python),
     *          the RV can be multi-dimensional, so,
     *          in order to translate this scipy.stats function,
     *          we implemented the Mvn class.
     *
     *          The Mvn class (MultiVariate Normal) is needed to return
     *          samples obtained by a multi-dimensional Gaussian function.
     */
    static Eigen::VectorXd markov_blanket(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale,
                                          double shape, double skewness);

    /**
     * @brief Returns the attributes of this family if using in a probabilistic model
     * @return A struct with attributes of the family
     *
     * @details Since the attributes link, mean_transform
     *          are np.array in the original code,
     *          we could not get what their purpose was;
     *          we translated them as y = x functions.
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
    static double neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale, double shape,
                                    double skewness);

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
    void vi_change_param(size_t index, double value) override;

    /**
     * @brief Wrapper function for selecting appropriate latent variable for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @return The appropriate indexed parameter
     */
    [[nodiscard]] double vi_return_param(size_t index) const override;

    /**
     * @brief Return the number of parameters
     * @return The number of parameters
     */
    [[nodiscard]] short unsigned int get_param_no() const override;

    /**
     * @brief Return the covariance prior
     * @return The covariance prior
     */
    [[nodiscard]] bool get_covariance_prior() const;

    /**
     * @brief Return the gradient of the location latent variable mu
     *  (used for variational inference)
     *  (python code works with both a single float and a vector of floats)
     * @param x A vector of random variables
     * @return The gradient of the location latent variable mu at x, for each variable
     *
     * @details The template implementation allows x to be a double,
     *          or a dinamic Eigen:Vector of doubles.
     *          Both definition and specialization are included in the .cpp file.
     */
    template<typename T>
    T vi_loc_score(const T& x) const;

    /**
     * @brief Return the score of the scale, where scale = exp(x)
     *  (used for variational inference)
     *  (python code works with both a single float and a vector of floats)
     * @param x A random variable, or a vector of random variables
     * @return The gradient of the scale latent variable at x, for each variable
     *
     * @details The template implementation allows x to be a double,
     *          or a dinamic Eigen:Vector of doubles.
     *          Both definition and specialization are included in the .cpp file.
     */
    // @Todo: chiedere a Busato
    template<typename T>
    T vi_scale_score(const T& x) const;

    /**
     * @brief Wrapper function for selecting appropriate score
     * @param x A random variable, or a vector of random variables
     * @param index 0 or 1 depending on which latent variable
     * @return The gradient of the scale latent variable at x
     */
     // @Todo: (NEW) perché è stata implementata qui?
    template<typename T>
    T vi_score(const T& x, size_t index) const {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, Eigen::VectorXd>);
        if (index == 0)
            return vi_loc_score(x);
        else if (index == 1)
            return vi_scale_score(x);
    }

    [[nodiscard]] std::string get_name() const override;

    [[nodiscard]] std::string get_z_name() const override;


    [[nodiscard]] Family* clone() const override; /**< override the family one,
 * returns a new Normal object by deep copy of the current one.
 */
};