#include "inference/bbvi_routines.hpp"

double covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // return (( x.array() - x.mean() ).cwiseProduct( y.array() - y.mean() ) ).mean();
    assert(x.size() == y.size());
    return static_cast<double>((x.array() - x.mean()).matrix().transpose() * (y.array() - y.mean()).matrix()) /
           (x.size() - 1);
}

void alpha_recursion(Eigen::VectorXd& alpha0, const Eigen::MatrixXd& grad_log_q, const Eigen::MatrixXd& gradient,
                     size_t param_no) {
    assert(grad_log_q.size() == grad_log_q.size());
    Eigen::Index lambda_i = 0;
    /*
     * the original file picks the covariance of two vectors xy
     */
    for (; lambda_i < param_no; lambda_i++)
        alpha0[lambda_i] = covariance(grad_log_q.row(lambda_i), gradient.row(lambda_i));
    return;
}

Eigen::VectorXd log_p_posterior(Eigen::MatrixXd& z,
                                const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& neg_posterior) {
    Eigen::Index i         = 0;
    Eigen::VectorXd result = Eigen::VectorXd::Zero(z.rows());

    for (; i < z.rows(); i++)
        result[i] = -neg_posterior(z.row(i), std::nullopt);

    return std::move(result);
}

Eigen::VectorXd mb_log_p_posterior(Eigen::MatrixXd& z,
                                   const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& neg_posterior,
                                   size_t mini_batch) {
    Eigen::Index i         = 0;
    Eigen::VectorXd result = Eigen::VectorXd::Zero(z.rows());

    for (; i < z.rows(); i++)
        result[i] = -neg_posterior(z.row(i), mini_batch);

    return std::move(result);
}
