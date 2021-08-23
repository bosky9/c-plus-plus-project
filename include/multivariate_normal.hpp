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
};