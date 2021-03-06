/**
 * @file multivariate_normal.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "multivariate_normal.hpp"

#include "Eigen/Core"        // Eigen::VectorXd, Eigen::MatrixXd, Eigen::Index
#include "Eigen/Eigenvalues" // Eigen::SelfAdjointEigenSolver

#include <algorithm> // std::transform, std::max
#include <chrono>    // std::chrono
#include <cmath>     // std::sqrt, std::pow, std::exp, std::erfc, std::expl, std::log
#include <random>    // std::default_random_engine, std::normal_distribution
#include <utility>   // std::move

Mvn::Mvn(Eigen::VectorXd mu, Eigen::MatrixXd s) : _mean{std::move(mu)}, _sigma{std::move(s)} {}

double Mvn::pdf(const Eigen::VectorXd& x) const {
    auto n          = static_cast<double>(x.size());
    double sqrt2pi  = std::sqrt(2 * M_PI);
    double quadform = (x - _mean).transpose() * _sigma.inverse() * (x - _mean);
    double norm     = std::pow(sqrt2pi, -n) * std::pow(_sigma.determinant(), -0.5);

    return norm * std::exp(-0.5 * quadform);
}

Eigen::VectorXd Mvn::sample(size_t nr_iterations) const {
    Eigen::Index n = _mean.size();

    // Generate x from the N(0, I) distribution
    Eigen::VectorXd x(n);
    Eigen::VectorXd sum(n);
    sum.setZero();
    for (size_t i{0}; i < nr_iterations; i++) {
        x.setRandom();
        x   = 0.5 * (x + Eigen::VectorXd::Ones(n));
        sum = sum + x;
    }
    sum = sum - (static_cast<double>(nr_iterations) / 2) * Eigen::VectorXd::Ones(n);
    x   = sum / (std::sqrt(static_cast<double>(nr_iterations) / 12));

    // Find the eigen vectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(_sigma);
    Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

    // Find the eigenvalues of the covariance matrix
    Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

    // Find the transformation matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
    Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
    Eigen::MatrixXd Q                = eigenvectors * sqrt_eigenvalues;

    return Q * x + _mean;
}

double Mvn::cdf(double x, [[maybe_unused]] double mean, [[maybe_unused]] double sigma) {
    return std::erfc(-x / std::sqrt(2)) / 2; // std::erfc is the Complementary Error Function
}

Eigen::VectorXd Mvn::logpdf(const Eigen::VectorXd& x, const Eigen::VectorXd& means, const Eigen::VectorXd& scales) {
    assert((means.size() == 1 && (scales.size() == 1 || x.size() == 1 || x.size() == scales.size())) ||
           (scales.size() == 1 && (x.size() == 1 || x.size() == means.size())) ||
           (means.size() == scales.size() && (x.size() == 1 || x.size() == means.size())));
    size_t size = std::max({x.size(), means.size(), scales.size()});
    Eigen::VectorXd result(size);
    const double ONE_OVER_SQRT_2PI{0.39894228040143267793994605993438};
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(size); ++i) {
        double x_val  = x.size() == 1 ? x(0) : x(i);
        double mean   = means.size() == 1 ? means(0) : means(i);
        double scale  = scales.size() == 1 ? scales(0) : scales(i);
        long double e = expl(-0.5 * std::pow((x_val - mean) / scale, 2.0));
        if (e == 0)
            result(i) = -(0.5 * std::pow((x_val - mean) / scale, 2.0)) + std::log(ONE_OVER_SQRT_2PI / scale);
        else
            result(i) = static_cast<double>(std::log((ONE_OVER_SQRT_2PI / scale) * e));
    }
    return result;
}

Eigen::MatrixXd Mvn::logpdf(const Eigen::MatrixXd& x, const Eigen::VectorXd& means, const Eigen::VectorXd& scales) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (Eigen::Index i{0}; i < x.rows(); i++)
        res.row(i) = logpdf(static_cast<Eigen::VectorXd>(x.row(i)), means, scales);
    return res;
}

Eigen::MatrixXd Mvn::logpdf(const Eigen::MatrixXd& x, const Eigen::MatrixXd& means, const Eigen::MatrixXd& scales) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (Eigen::Index i{0}; i < x.rows(); i++)
        res.row(i) = logpdf(static_cast<Eigen::VectorXd>(x.row(i)), static_cast<Eigen::VectorXd>(means.row(i)),
                            static_cast<Eigen::VectorXd>(scales.row(i)));
    return res;
}

Eigen::VectorXd Mvn::random(double mean, double scale, size_t n) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution{mean, scale};
    Eigen::VectorXd rands(n);
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(n); ++i) {
        rands(i) = distribution(generator);
    }
    return rands;
}

Eigen::VectorXd Mvn::random(const Eigen::VectorXd& mean, double scale, size_t n) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    Eigen::VectorXd rands(n);
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(n); ++i) {
        std::normal_distribution<double> distribution{mean(i), scale};
        rands(i) = distribution(generator);
    }
    return rands;
}

Eigen::MatrixXd Mvn::random(const Eigen::VectorXd& mean, const Eigen::VectorXd& scale, size_t n) {
    assert(mean.size() == scale.size());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    Eigen::MatrixXd rands(n, mean.size());
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(n); ++i) {
        for (Eigen::Index j{0}; j < mean.size(); j++) {
            std::normal_distribution<double> distribution{mean(j), scale(j)};
            rands(i, j) = distribution(generator);
        }
    }
    return rands;
}