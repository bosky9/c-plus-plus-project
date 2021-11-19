#pragma once

#define _USE_MATH_DEFINES

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd

/**
 * @class Mvn multivariate_normal.hpp
 * @brief Class that represents a multivariate normal distribution
 */
class Mvn final {
public:
    Eigen::VectorXd _mean;
    Eigen::MatrixXd _sigma;

    /**
     * @brief Constructor for Multivariate Normal distribution
     * @param mu Vector of means
     * @param s Matrix of scales
     */
    Mvn(Eigen::VectorXd mu, const Eigen::MatrixXd& s);

    /**
     * @brief Compute the PDF (Probabilty Density Function) of the distribution
     * @param x Vector of indices
     * @return PDF value
     */
    [[nodiscard]] double pdf(const Eigen::VectorXd& x) const;

    /**
     * @brief Returns a random sample from the distribution
     * @param nr_iterations Number of iterations
     * @return Random sample from the distribution
     */
    [[nodiscard]] Eigen::VectorXd sample(size_t nr_iterations = 20) const;

    // The following field are used as mathematical formulas, and are independent from the class itself.

    /**
     * @brief Compute the CDF (Cumulative Distribution Function) of the distribution (like scipy.stats.norm.cdf)
     * @param x Value
     * @return CDF value
     */
    [[nodiscard]] static double cdf(double x, double mean = 0.0, double sigma = 1.0);

    /**
     * @brief Compute the PDF of a normal distribution (like scipy.stats.norm.pdf)
     * @param x Vector of indices
     * @param mean Mean
     * @param sigma Scale
     * @return Vector of PDF values in x
     */
    static Eigen::VectorXd pdf(const Eigen::VectorXd& x, double mean = 0.0, double sigma = 1.0);

    /**
     * @brief Compute the logpdf of a normal distribution (like scipy.stats.norm.logpdf)
     * @param x Vector of indexes
     * @param means Vector of means of the normal distribution
     * @param scales Vector of scales of the normal distribution
     * @return Vector of logpdf values in x
     *
     * @details     This is the numpy case where the logpdf is computed for every x,
     *              using the same mean and variance for each x.
     *
     *              at row 68 "e = expl(-0.5 * pow((x_val - mean) / scale, 2.0));"
     *              there may be an overflow, because of the huge size of having e(-[something big]).
     *
     *              However, since we are interested in the natural logarithm of this value,
     *              we can ignore the exp and simply sum its content to log(ONE_OVER_SQRT_2PI / scale).
     *              Remember that log(a*b) = log(a) + log(b).
     *
     *              Basically:
     *              log(e(a)*b) = log(e(a)) + log(b) = a + log(b).
     *              The overflow is then avoided.
     *
     */
    [[nodiscard]] static Eigen::VectorXd logpdf(const Eigen::VectorXd& x, const Eigen::VectorXd& means,
                                                const Eigen::VectorXd& scales);

    /**
     * @brief Compute the logpdf of a normal distribution (like scipy.stats.norm.logpdf)
     * @param x Matrix of indexes
     * @param means Vector of means of the normal distribution
     * @param scales Vector of scales of the normal distribution
     * @return Matrix of logpdf values in x
     *
     * @details     This is the numpy case where
     *              each row of x values uses the same means vector and variances vector,
     *              in order to compute the logpdf of each x.
     */
    [[nodiscard]] static Eigen::MatrixXd logpdf(const Eigen::MatrixXd& x, const Eigen::VectorXd& means,
                                                const Eigen::VectorXd& scales);

    /**
     * @brief Compute the logpdf of a normal distribution (like scipy.stats.norm.logpdf)
     * @param x Matrix of indexes
     * @param means Matrix of means of the normal distribution
     * @param scales Matrix of scales of the normal distribution
     * @return Matrix of logpdf values in x
     *
     * @details     This is the numpy case where
     *              each row of x values uses its own means vector and variances vector,
     *              in order to compute the logpdf of each x.
     */
    [[nodiscard]] static Eigen::MatrixXd logpdf(const Eigen::MatrixXd& x, const Eigen::MatrixXd& means,
                                                const Eigen::MatrixXd& scales);

    /**
     * @brief Generate random values from a normal distribution
     * @param mean Mean of the normal distribution
     * @param scale Scale of the normal distribution
     * @param n Number of values to generate
     * @return Vector of random values
     */
    [[nodiscard]] static Eigen::VectorXd random(double mean, double scale, size_t n);

    /**
     * @brief Generate random values from a normal distribution
     * @param mean Means of the normal distribution
     * @param scale Scale of the normal distribution
     * @param n Number of values to generate
     * @return Vector of random values
     */
    [[nodiscard]] static Eigen::VectorXd random(const Eigen::VectorXd& mean, double scale, size_t n);

    /**
     * @brief Generate random values from a normal distribution
     * @param mean Means of the normal distributions
     * @param scale Scales of the normal distributions
     * @param n Number of values to generate
     * @return Vector of random values
     */
    [[nodiscard]] static Eigen::MatrixXd random(const Eigen::VectorXd& mean, const Eigen::VectorXd& scale, size_t n);
};