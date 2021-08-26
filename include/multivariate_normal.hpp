#pragma once

#define _USE_MATH_DEFINES

#include <Eigen/Eigenvalues>
#include <cmath>

class Mvn {
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
    static Eigen::VectorXd logpdf(const Eigen::VectorXd& x, const Eigen::VectorXd& means,
                                  const Eigen::VectorXd& scales);

    static Eigen::MatrixXd logpdf(const Eigen::MatrixXd& x, const Eigen::VectorXd& means,
                                  const Eigen::VectorXd& scales);

    static Eigen::MatrixXd logpdf(const Eigen::MatrixXd& x, const Eigen::MatrixXd& means,
                                  const Eigen::MatrixXd& scales);

    /**
     * @brief Generate random values from a normal distribution
     * @param mean Mean of the normal distribution
     * @param scale Scale of the normal distribution
     * @param n Number of values to generate
     * @return Vector of random values
     */
    static Eigen::VectorXd random(double mean, double scale, size_t n);
};