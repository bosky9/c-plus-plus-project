/**
 * @file test_norm_post_sim.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "inference/norm_post_sim.hpp"

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include <catch2/catch_test_macros.hpp>
#include "inference/sample.hpp" // Sample

TEST_CASE("Check sizes for norm_post_sim", "[norm_post_sim]") {
    Eigen::VectorXd modes{Eigen::VectorXd::Zero(3)};
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(3, 3)};
    Sample data = nps::norm_post_sim(modes, cov_matrix);

    REQUIRE(data.chain.cols() == 30000);
    REQUIRE(data.chain.rows() == 3);
    REQUIRE(data.mean_est.size() == 3);
    REQUIRE(data.median_est.size() == 3);
    REQUIRE(data.upper_95_est.size() == 3);
    REQUIRE(data.lower_95_est.size() == 3);
}

TEST_CASE("Check normal distribution for norm_post_sim", "[norm_post_sim]") {
    Eigen::Index size{3};
    Eigen::VectorXd modes{Eigen::VectorXd::Zero(size)};
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(size, size)};
    Sample data = nps::norm_post_sim(modes, cov_matrix);

    for (int64_t i{0}; i < size; ++i) {
        REQUIRE((data.mean_est[i] > -0.1 && data.mean_est[i] < 0.1));
        REQUIRE((data.median_est[i] > -0.2 && data.median_est[i] < 0.2));
        REQUIRE((data.upper_95_est[i] > 1));
        REQUIRE((data.lower_95_est[i] < -1));
    }
}