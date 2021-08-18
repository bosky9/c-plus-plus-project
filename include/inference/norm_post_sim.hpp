#include <algorithm>
#include <multivariate_normal.hpp>

#include "headers.hpp"

/**
 * @brief Number of simulations
 */
const size_t NSIMS = 30000;

// Chiedere al prof
// 1 dispari    (true)
// 0 pari       (false)
/**
 * @brief Compile time function that returns if NSIMS is odd
 */
constexpr bool NSIMS_ODD() {
    return static_cast<bool>(NSIMS % 2);
};

/**
 * @brief Compile time function that returns the index representing the 95 percentile
 */
constexpr size_t NSIMS_95() {
    return NSIMS * 95 / 100;
};

/**
 * @brief Compile time function that returns the index representing the 5 percentile
 */
constexpr size_t NSIMS_5() {
    return NSIMS * 5 / 100;
};

/**
 * @brief Data returned by norm_post_sim function
 */
struct NormPostSimData {
    Eigen::Matrix<double, Eigen::Dynamic, NSIMS> chain;
    std::vector<double> mean_est;
    std::vector<double> median_est;
    std::vector<double> upper_95_est;
    std::vector<double> lower_95_est;
};

/**
 * @brief
 * @param modes Modes vector
 * @param cov_matrix Covariances matrix
 * @return A NormPostSimData structure
 */
NormPostSimData norm_post_sim(const Eigen::VectorXd& modes, const Eigen::MatrixXd& cov_matrix) {
    Mvn mul_norm{modes, cov_matrix};
    Eigen::Matrix<double, NSIMS, Eigen::Dynamic> phi =
            Eigen::Matrix<double, NSIMS, Eigen::Dynamic>::Zero(NSIMS, modes.size());

    for (size_t i{0}; i < NSIMS; i++) {
        phi[i] = mul_norm.sample(); // Incollo un vettore di lunghezza len(modes)
    }

    NormPostSimData data{};
    data.chain = phi.transpose();

    data.mean_est = dynamic_cast<std::vector<double>>(phi.colwise().mean());

    for (auto col : phi.colwise()) {
        auto col_sort = std::sort(dynamic_cast<std::vector<double>>(col));

        if (NSIMS_ODD)
            data.median_est.push_back(col_sort[NSIMS / 2]);
        else
            data.median_est.push_back((col_sort[NSIMS / 2] + col_sort.at[NSIMS / 2 - 1]) / 2);

        data.upper_95_est.push_back((col_sort[NSIMS_95]));
        data.lower_95_est.push_back((col_sort[NSIMS_5]));
    }

    return data;
}