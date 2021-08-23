#include "headers.hpp"

/**
 * @brief This function is used inside the BBVI classes
 * @param alpha0
 * @param grad_log_q
 * @param gradient
 * @param param_no
 * @return
 */
std::vector<double> alpha_recursion(
        std::vector<double>& alpha0,
        Eigen::MatrixXd& grad_log_q,
        Eigen::MatrixXd& gradient,
        int param_no);

/**
 * @brief This function is used inside the BBVI classes
 * @param alpha0
 * @param grad_log_q
 * @param gradient
 * @param param_no
 * @return
 */
std::vector<double> log_p_posterior(
        Eigen::MatrixXd& z,
        const std::function<double(Eigen::VectorXd)>& neg_posterior
        );

/**
 * @brief This function is used inside the BBVI classes
 * @param alpha0
 * @param grad_log_q
 * @param gradient
 * @param param_no
 * @return
 */
std::vector<double> mb_log_p_posterior(
        Eigen::MatrixXd& z,
        const std::function<double(Eigen::VectorXd)>& neg_posterior,
        int mini_batch
    );
