/**
 * @file normal.hpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include "families/family.hpp"

#include <cassert> // static_assert(), assert()

/**
 * @class Normal normal.hpp
 * @brief Normal distribution for time series
 *
 * @details Since this project won't cover GAS models, the involved chunk of code has not been translated.
 */
class Normal final : public Family {
public:
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
     * @param h_approx Variance of the measurement density
     * @param data Univariate time series data (define the size of the created matrices)
     * @return A pointer to an array consisting of two Eigen::MatrixXd:
     *  - H: approximating measurement variance matrix
     *  - mu: approximating measurement constants
     *
     *  @details Parameters beta, T, Z, R, Q were present but not actually used
     */
    [[nodiscard]] static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> approximating_model(double h_approx,
                                                                                         const Eigen::VectorXd& data);

    /**
     * @brief Creates approximating Gaussian state space model for the Normal measurement density.
     * @param h_approx Variance of the measurement density
     * @param data Univariate time series data (define the size of the created matrices)
     * @return A pointer to an array consisting of two Eigen::MatrixXd:
     *  - H: approximating measurement variance matrix
     *  - mu: approximating measurement constants
     *
     *  @details Parameters beta, T, Z, R, Q, X, state_no were present but not actually used
     */
    [[nodiscard]] static std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    approximating_model_reg(double h_approx, const Eigen::VectorXd& data);

    /**
     * @brief Builds additional latent variables for this family in a probabilistic model
     * @return A list of structs (each struct contains latent variable information)
     */
    [[nodiscard]] std::vector<lv_to_build> build_latent_variables() const override;

    /**
     * @brief Draws random variables from this distribution with new latent variables
     * @param loc Location parameter(s) for the distribution
     * @param scale Scale parameter for the distribution
     * @param nsims Number of draws to take from the distribution
     * @return Random draws from the distribution, obtained thanks to the std::normal_distribution library.
     *
     * @details Parameters shape and skewness were present but not actually used
     */
    [[nodiscard]] Eigen::VectorXd draw_variable(double loc, double scale, int64_t nsims) const override;

    [[nodiscard]] Eigen::VectorXd draw_variable(const Eigen::VectorXd& loc, double scale, int64_t nsims) const override;

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param size How many simulations to perform
     * @return Array of Normal random variable
     */
    [[nodiscard]] Eigen::VectorXd draw_variable_local(int64_t size) const override;

    /**
     * @brief Log PDF for Normal prior
     * @param mu Latent variable for which the prior is being formed over
     * @return log(p(mu))
     */
    [[nodiscard]] double logpdf(double mu) const override;

    /**
     * @brief Markov blanket for each likelihood term - used for space state models
     * @param y Univariate time series
     * @param mean Array of location parameters for the Normal distribution
     * @param scale Scale parameter for the Normal distribution
     * @return Markov blanket of the Normal family
     *
     * @details The original Python code is
     *
     *          return ss.norm.logpdf(y, loc=mean, scale=scale)
     *
     *          Which returns the log of the probability density function at y of the given RV.
     *          Since mean can be a vector (in Python), the RV can be multi-dimensional, so, in order to translate this
     *          scipy.stats function,we implemented the Mvn class.
     *
     *          The Mvn class (MultiVariate Normal) is needed to return samples obtained by a multi-dimensional Gaussian
     *          function.
     *
     *          (Parameters shape and skewness were present but not actually used.)
     */
    static Eigen::VectorXd markov_blanket(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale);

    /**
     * @brief Returns the attributes of this family if using in a probabilistic model
     * @return A struct with attributes of the family
     *
     * @details Since the attributes link, mean_transform are np.array in the original code,
     *          we translated them as y = x functions.
     */
    [[nodiscard]] FamilyAttributes setup() const override;

    /**
     * @brief Negative loglikelihood function for this distribution
     * @param y Univariate time series
     * @param mean Array of location parameters for the Normal distribution
     * @param scale Scale parameter for the Normal distribution
     * @return Negative loglikelihood of the Normal family
     *
     * @details Parameters shape and skewness were present but not actually used.
     */
    [[nodiscard]] double neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean,
                                           double scale) const override;

    /**
     * @brief PDF for Normal prior
     * @param mu Latent variable for which the prior is being formed over
     * @return p(mu)
     */
    [[nodiscard]] double pdf(double mu);

    /**
     * @brief Wrapper function for changing latent variables for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @param value What to change the latent variable to
     */
    void vi_change_param(uint8_t index, double value) override;

    /**
     * @brief Wrapper function for selecting appropriate latent variable for variational inference
     * @param index 0 or 1 depending on which latent variable
     * @return The appropriate indexed parameter
     */
    [[nodiscard]] double vi_return_param(uint8_t index) const override;

    /**
     * @brief Return the gradient of the location latent variable mu
     *  (used for variational inference)
     *  (Python code works with both a single float and a vector of floats)
     * @param x A vector of random variables
     * @return The gradient of the location latent variable mu at x, for each variable
     *
     * @details In Python, x could be a float or an array; if x was an array, the code would work anyway,
     *          using element-wise operations.
     *
     *          The template implementation allows x to be a double, or a dynamic Eigen:Vector of doubles.
     *          Both definition and specialization are included in the .cpp file.
     *
     *          Notice how, in order to implement element-wise operations, we need to cast x as an Eigen:Array:
     *          ... x.array() + 4 ... adds 4 to each element of x.
     */
    template<typename T>
    T vi_loc_score(const T& x) const {
        return {};
    }

    /**
     * @brief Return the score of the scale, where scale = exp(x)
     *  (used for variational inference)
     *  (Python code works with both a single float and a vector of floats)
     * @param x A random variable, or a vector of random variables
     * @return The gradient of the scale latent variable at x, for each variable
     *
     * @details In Python, x could be a float or an array; if x was an array, the code would work anyway,
     *          using element-wise operations.
     *
     *          The template implementation allows x to be a double, or a dynamic Eigen:Vector of doubles.
     *          Both definition and specialization are included in the .cpp file.
     *
     *          Notice how, in order to implement element-wise operations, we need to cast x as an Eigen:Array:
     *          ... x.array() + 4 ... adds 4 to each element of x.
     */
    template<typename T>
    T vi_scale_score(const T& x) const {
        return {};
    }

    /**
     * @brief Wrapper function for selecting appropriate score
     * @param x A random variable, or a vector of random variables
     * @param index 0 or 1 depending on which latent variable
     * @return The gradient of the scale latent variable at x
     */
    template<typename T>
    T vi_score(const T& x, uint8_t index) const {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, Eigen::VectorXd>,
                      "Variable must be a double or an Eigen::VectorXd");
        assert((index == 0 || index == 1) && "Index is neither 0 nor 1");

        if (index == 0)
            return vi_loc_score(x);
        else if (index == 1)
            return vi_scale_score(x);
        return {};
    }


    // Get methods -----------------------------------------------------------------------------------------------------

    /**
     * @brief Returns the mean of the distribution
     * @return Mean of the distribution
     */
    [[nodiscard]] double get_mu0() const;

    /**
     * @brief Get the name of the distribution family for the get_z_priors_names() method of LatentVariables
     * @return Name of the distribution family
     */
    [[nodiscard]] std::string get_name() const override;

    /**
     * @brief Return the number of parameters
     * @return The number of parameters
     */
    [[nodiscard]] uint8_t get_param_no() const override;

    /**
     * @brief Returns the variance of the distribution
     * @return Variance of the distribution
     */
    [[nodiscard]] double get_sigma0() const;

    /**
     * @brief Get the description of the parameters of the distribution family for the get_z_priors_names() method of
     * LatentVariables
     * @return Description of the parameters of the distribution family
     */
    [[nodiscard]] std::string get_z_name() const override;

    // Clone function --------------------------------------------------------------------------------------------------

    /**
     * @brief Returns a clone of the current object
     * @return A copy of the family object which calls this function
     *
     * @details Overrides the family one, returns a new Normal object by deep copy of the current one.
     */
    [[nodiscard]] Family* clone() const override;

private:
    double _mu0;                  ///< The mean of the Gaussian
    double _sigma0;               ///< The variance of the Gaussian
    short unsigned int _param_no; ///< Number of parameters
    bool _covariance_prior;       ///< Covariance's prior
    // gradient_only won't be used (no GAS models)
};