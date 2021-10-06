#pragma once

#define _USE_MATH_DEFINES

#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>
#include <random>

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
    [[nodiscard]] Eigen::VectorXd sample(unsigned int nr_iterations = 20) const;

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

    // @TODO: Consider a template approach
    /**
     * @brief Compute the logpdf of a normal distribution (like scipy.stats.norm.logpdf)
     * @param x Vector of indexes
     * @param means Vector of means of the normal distribution
     * @param scales Vector of scales of the normal distribution
     * @return Vector of logpdf values in x
     */
    [[nodiscard]] static Eigen::VectorXd logpdf(const Eigen::VectorXd& x, const Eigen::VectorXd& means,
                                                const Eigen::VectorXd& scales);

    [[nodiscard]] static Eigen::MatrixXd logpdf(const Eigen::MatrixXd& x, const Eigen::VectorXd& means,
                                                const Eigen::VectorXd& scales);

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
     * @param mean Means of the normal distributions
     * @param scale Scales of the normal distributions
     * @param n Number of values to generate
     * @return Vector of random values
     */
    [[nodiscard]] static Eigen::MatrixXd random(Eigen::VectorXd mean, Eigen::VectorXd scale, size_t n);
};