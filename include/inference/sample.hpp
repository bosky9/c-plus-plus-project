#pragma once

/**
 * @brief Sample data returned by many functions
 */
struct Sample final {
    Eigen::MatrixXd chain;        ///< Chains for each parameter
    Eigen::VectorXd mean_est;     ///< Mean values for each parameter
    Eigen::VectorXd median_est;   ///< Median values for each parameter
    Eigen::VectorXd upper_95_est; ///< Upper 95% credibility interval for each parameter
    Eigen::VectorXd lower_95_est; ///< Lower 95% credibility interval for each parameter
};