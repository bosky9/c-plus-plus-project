/**
 * @file bbvi_routines.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/bbvi_routines.hpp"

double bbvi_routines::covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    assert(x.size() == y.size());
    return static_cast<double>((x.array() - x.mean()).matrix().transpose() * (y.array() - y.mean()).matrix()) /
           static_cast<double>(x.size() - 1);
}

void bbvi_routines::alpha_recursion(Eigen::VectorXd& alpha0, const Eigen::MatrixXd& grad_log_q,
                                    const Eigen::MatrixXd& gradient, uint8_t param_no) {
    assert(grad_log_q.size() == gradient.size());
    for (Eigen::Index lambda_i{0}; lambda_i < param_no; ++lambda_i)
        alpha0[lambda_i] = bbvi_routines::covariance(grad_log_q.row(lambda_i), gradient.row(lambda_i));
}

Eigen::VectorXd bbvi_routines::log_p_posterior(
        const Eigen::MatrixXd& z,
        const std::function<double(const Eigen::VectorXd&, std::optional<size_t>)>& neg_posterior) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(z.rows());

    for (Eigen::Index i{0}; i < z.rows(); ++i)
        result[i] = -neg_posterior(z.row(i), std::nullopt);

    return result;
}

Eigen::VectorXd bbvi_routines::mb_log_p_posterior(
        const Eigen::MatrixXd& z,
        const std::function<double(const Eigen::VectorXd&, std::optional<size_t>)>& neg_posterior, size_t mini_batch) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(z.rows());

    for (Eigen::Index i{0}; i < z.rows(); ++i)
        result[i] = -neg_posterior(z.row(i), mini_batch);

    return result;
}
