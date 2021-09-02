/**
 * @brief Sample data returned by many functions
 */
struct Sample final {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> chain; ///< Chains for each parameter
    std::vector<double> mean_est;                                ///< Mean values for each parameter
    std::vector<double> median_est;                              ///< Median values for each parameter
    std::vector<double> upper_95_est;                            ///< Upper 95% credibility interval for each parameter
    std::vector<double> lower_5_est;                             ///< Lower 95% credibility interval for each parameter
};