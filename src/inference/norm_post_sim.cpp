/**
 * @file norm_post_sim.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/norm_post_sim.hpp"

#include "multivariate_normal.hpp" // Mvn

Sample nps::norm_post_sim(const Eigen::VectorXd& modes, const Eigen::MatrixXd& cov_matrix) {
    Mvn mul_norm{modes, cov_matrix};
    Eigen::Index modes_len = modes.size();
    Eigen::Matrix<double, nps::NSIMS, Eigen::Dynamic> phi =
            Eigen::Matrix<double, nps::NSIMS, Eigen::Dynamic>::Zero(nps::NSIMS, modes_len);

    for (Eigen::Index i{0}; i < NSIMS; ++i) {
        phi.row(i) = mul_norm.sample();
    }

    Eigen::MatrixXd chain    = phi.transpose();
    Eigen::VectorXd mean_est = phi.colwise().mean();

    std::vector<double> median_est{}, upper_95_est{}, lower_5_est{};
    for (int64_t i{0}; i < modes_len; ++i) {
        Eigen::VectorXd col_sort{phi.col(i)};
        std::sort(col_sort.begin(), col_sort.end());

        if (nps::NSIMS_ODD)
            median_est.push_back(col_sort[nps::NSIMS / 2]);
        else
            median_est.push_back((col_sort[nps::NSIMS / 2] + col_sort[nps::NSIMS / 2 - 1]) / 2);

        upper_95_est.push_back(col_sort[nps::N95]);
        lower_5_est.push_back(col_sort[nps::N5]);
    }

    return {chain, mean_est, Eigen::VectorXd::Map(&median_est[0], static_cast<Eigen::Index>(median_est.size())),
            Eigen::VectorXd::Map(&upper_95_est[0], static_cast<Eigen::Index>(upper_95_est.size())),
            Eigen::VectorXd::Map(&lower_5_est[0], static_cast<Eigen::Index>(lower_5_est.size()))};
}