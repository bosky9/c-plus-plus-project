/**
 * @brief Sample data returned by many functions
 */
struct Sample {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> chain;
    std::vector<double> mean_est;
    std::vector<double> median_est;
    std::vector<double> upper_95_est;
    std::vector<double> lower_5_est;
};