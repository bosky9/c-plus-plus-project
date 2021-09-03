#pragma once

#define _USE_MATH_DEFINES

#include <Eigen/Eigenvalues>
#include <cmath>

class Mvn final {
public:
    Eigen::VectorXd _mean;
    Eigen::MatrixXd _sigma;

    Mvn(const Eigen::VectorXd& mu, const Eigen::MatrixXd& s);
    [[nodiscard]] double pdf(const Eigen::VectorXd& x) const;
    [[nodiscard]] Eigen::VectorXd sample(unsigned int nr_iterations = 20) const;

    // @Todo: consider a template approach
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