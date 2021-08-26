#include "multivariate_normal.hpp"

#include <chrono>
#include <random>

Mvn::Mvn(const Eigen::VectorXd& mu, const Eigen::MatrixXd& s) : _mean{mu}, _sigma{s} {}

double Mvn::pdf(const Eigen::VectorXd& x) const {
    double n        = static_cast<double>(x.rows());
    double sqrt2pi  = std::sqrt(2 * M_PI);
    double quadform = (x - _mean).transpose() * _sigma.inverse() * (x - _mean);
    double norm     = std::pow(sqrt2pi, -n) * std::pow(_sigma.determinant(), -0.5);

    return norm * exp(-0.5 * quadform);
}

Eigen::VectorXd Mvn::sample(unsigned int nr_iterations) const {
    Eigen::Index n = _mean.rows();

    // Generate x from the N(0, I) distribution
    Eigen::VectorXd x(n);
    Eigen::VectorXd sum(n);
    sum.setZero();
    for (size_t i = 0; i < nr_iterations; i++) {
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

Eigen::VectorXd Mvn::logpdf(const Eigen::VectorXd& x, const Eigen::VectorXd& means, const Eigen::VectorXd& scales) {
    assert((means.size() == 1 && (scales.size() == 1 || x.size() == 1 || x.size() == scales.size())) ||
           (scales.size() == 1 && (x.size() == 1 || x.size() == means.size())) ||
           (means.size() == scales.size() && (x.size() == 1 || x.size() == means.size())));
    size_t size = std::max({x.size(), means.size(), scales.size()});
    Eigen::VectorXd result(size);
    const double ONE_OVER_SQRT_2PI = 0.39894228040143267793994605993438;
    for (Eigen::Index i{0}; i < size; i++) {
        double x_val = x.size() == 1 ? x(0) : x(i);
        double mean  = means.size() == 1 ? means(0) : means(i);
        double scale = scales.size() == 1 ? scales(0) : scales(i);
        result(i)    = log((ONE_OVER_SQRT_2PI / scale) * exp(-0.5 * pow((x_val - mean) / scale, 2.0)));
    }
    return result;
}

Eigen::MatrixXd Mvn::logpdf(const Eigen::MatrixXd& x, const Eigen::VectorXd& means, const Eigen::VectorXd& scales) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); i++)
        res.row(i) = logpdf(static_cast<Eigen::VectorXd>(x.row(i)), means, scales);
    return res;
}

Eigen::MatrixXd Mvn::logpdf(const Eigen::MatrixXd& x, const Eigen::MatrixXd& means, const Eigen::MatrixXd& scales) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); i++)
        res.row(i) = logpdf(static_cast<Eigen::VectorXd>(x.row(i)), static_cast<Eigen::VectorXd>(means.row(i)),
                            static_cast<Eigen::VectorXd>(scales.row(i)));
    return res;
}

Eigen::VectorXd Mvn::random(double mean, double scale, size_t n) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution{mean, scale};
    Eigen::VectorXd rands(n);
    for (Eigen::Index i{0}; i < n; i++) {
        rands(i) = distribution(generator);
    }
    return rands;
}