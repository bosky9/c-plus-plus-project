#include "inference/bbvi_routines.hpp"

std::vector<double> alpha_recursion(std::vector<double>& alpha0, Eigen::MatrixXd& grad_log_q, Eigen::MatrixXd& gradient,
        int param_no) {
    size_t lambda_i;
    for(; lambda_i < param_no; lambda_i++) {
        Eigen::MatrixXd centered = mat.rowwise() - mat.colwise().mean();
        Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(mat.rows() - 1);
        alpha0[lambda_i] =
    }
}

std::vector<double> log_p_posterior(
        Eigen::MatrixXd& z,
        const std::function<double(Eigen::VectorXd)>& neg_posterior
        ) {

}

std::vector<double> mb_log_p_posterior(
        Eigen::MatrixXd& z,
        const std::function<double(Eigen::VectorXd)>& neg_posterior,
        int mini_batch
        ) {

}

