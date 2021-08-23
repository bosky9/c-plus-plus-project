#include "inference/bbvi_routines.hpp"

std::vector<double> alpha_recursion(std::vector<double>& alpha0, Eigen::MatrixXd& grad_log_q, Eigen::MatrixXd& gradient,
        size_t param_no) {
    Eigen::Index lambda_i;
    /*
    * the original file picks the covariance of two vectors xy
    */
    for(; lambda_i < param_no; lambda_i++)
        alpha0[lambda_i] = ((grad_log_q.row(lambda_i).array() - grad_log_q.row(lambda_i).mean()).cwiseProduct(gradient.row(lambda_i).array() - gradient.row(lambda_i).mean())).mean();
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

