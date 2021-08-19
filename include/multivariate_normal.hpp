#include <Eigen/Eigenvalues>

#include "headers.hpp"

class Mvn {
public:
    Mvn(const Eigen::VectorXd& mu, const Eigen::MatrixXd& s);
    [[nodiscard]] double pdf(const Eigen::VectorXd& x) const;
    [[nodiscard]] Eigen::VectorXd sample(unsigned int nr_iterations = 20) const;
    Eigen::VectorXd _mean;
    Eigen::MatrixXd _sigma;
};