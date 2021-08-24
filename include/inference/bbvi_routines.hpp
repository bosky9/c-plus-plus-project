#include "headers.hpp"

double covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

/**
 * @brief This function is used inside the BBVI classes
 * @param alpha0
 * @param grad_log_q
 * @param gradient
 * @param param_no
 * @return
 */
void alpha_recursion(Eigen::VectorXd& alpha0, const Eigen::MatrixXd& grad_log_q, const Eigen::MatrixXd& gradient,
                     size_t param_no);

/**
 * @brief This function is used inside the BBVI classes
 * @param z A matrix from which to extract the vectors (rows) used in neg_posterior
 * @param neg_posterior A function which takes a vector (Eigen::VectorXd) and returns a double
 * @return An array of doubles (using std::move())
 */
Eigen::VectorXd log_p_posterior(Eigen::MatrixXd& z, const std::function<double(Eigen::VectorXd)>& neg_posterior);

/**
 * @brief This function is used inside the BBVI classes
 * @param alpha0
 * @param grad_log_q
 * @param gradient
 * @param param_no
 * @return
 */
Eigen::VectorXd mb_log_p_posterior(Eigen::MatrixXd& z, const std::function<double(Eigen::VectorXd, int)>& neg_posterior,
                                   int mini_batch);
