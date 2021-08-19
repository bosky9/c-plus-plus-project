#include <algorithm>
#include <multivariate_normal.hpp>

#include "headers.hpp"

/**
 * @brief Number of simulations
 */
const int NSIMS = 30000;

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
constexpr int NSIMS_95(int nsims) {
    return static_cast<int>(nsims / (100 / 95));
};

/**
 * @brief Compile time function that returns the index representing the 5 percentile
 */
constexpr int NSIMS_5(int nsims) {
    return static_cast<int>(nsims / (100 / 5));
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
    int modes_len = modes.size();
    Eigen::Matrix<double, NSIMS, Eigen::Dynamic> phi =
            Eigen::Matrix<double, NSIMS, Eigen::Dynamic>::Zero(NSIMS, modes_len);

    for (size_t i{0}; i < NSIMS; i++) {
        phi.row(i) = mul_norm.sample(); // Incollo un vettore di lunghezza modes.size() in ogni riga
    }

    NormPostSimData data{};
    data.chain = phi.transpose();

    Eigen::VectorXd mean_vector = phi.colwise().mean();
    data.mean_est.resize(mean_vector.size());
    Eigen::VectorXd::Map(&data.mean_est[0], mean_vector.size()) = mean_vector;

    /*for (int i = 0; i < modes_len; i++)
        data.mean_est.push_back(mean_vector[i]); */

    //data.mean_est = dynamic_cast<std::vector<double>>(phi.colwise().mean());

    for (int i = 0; i < modes_len; i++) {
        std::vector<double> col_sort(NSIMS);
        Eigen::VectorXd::Map(&col_sort[0], NSIMS) = phi.col(i);
        std::sort(col_sort.begin(), col_sort.end());

        if (NSIMS_ODD)
            data.median_est.push_back(col_sort[NSIMS / 2]);
        else
            data.median_est.push_back((col_sort[NSIMS / 2] + col_sort[NSIMS / 2 - 1]) / 2);

        data.upper_95_est.push_back((col_sort[NSIMS_95(NSIMS)]));
        data.lower_95_est.push_back((col_sort[NSIMS_5(NSIMS)]));
    }

    return data;
}