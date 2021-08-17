#include "headers.hpp"

struct NormPostSimData {
    Eigen::Matrix<double, Dynamic, 30000> chain;
    std::vector<double> mean_est;
    std::vector<double> median_est;
    std::vector<double> upper_95_est;
    std::vector<double> lower_95_est;
};

NormPostSimData norm_post_sim(const Eigen::VectorXd& modes, const Eigen::MatrixXd& cov_matrix) {
    Mvn mul_norm{modes, cov_matrix};
    size_t nsims = 30000;
    Eigen::Matrix<double, 30000, Dynamic> phi = Eigen::Eigen::Matrix<double, 30000, Dynamic>::Zero(30000, modes.size


    for (size_t i{0}; i < nsims; i++) {
        phi[i] = mul_norm.sample();
    }

    Eigen::Matrix<double, Dynamic, 30000> chain = phi.transpose();

}