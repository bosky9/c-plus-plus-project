#include "hessian.hpp"

using namespace derivatives;

Eigen::MatrixXd derivatives::hessian(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v) {
    Eigen::MatrixXd H(v.size(), v.size());
    for (Eigen::Index i{0}; i < v.size(); ++i) {
        for (Eigen::Index j = i; j < v.size(); j++) {
            H(i, j) = second_derivative(function, v, i, j);
            H(j, i) = H(i, j);
        }
    }
    return std::move(H);
}

double derivatives::first_derivative(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v,
                                     Eigen::Index i) {
    v[i] -= h;
    double fl{function(v)};
    v[i] += 2 * h;
    double fr{function(v)};
    v[i] -= h;
    return (fr - fl) / (2 * h);
}

double derivatives::second_derivative(const std::function<double(Eigen::VectorXd)>& function, Eigen::VectorXd& v,
                                      Eigen::Index i, Eigen::Index j) {
    v[j] -= h;
    double fl{first_derivative(function, v, i)};
    v[j] += 2 * h;
    double fr{first_derivative(function, v, i)};
    v[j] -= h;
    return (fr - fl) / (2 * h);
}