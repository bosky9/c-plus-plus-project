#include <catch2/catch_test_macros.hpp>

#include "inference/norm_post_sim.hpp"

TEST_CASE("norm_post_sim check sizes", "[norm_post_sim]") {
    Eigen::VectorXd modes = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(3, 3)};
    Sample data = norm_post_sim(modes, cov_matrix);

    REQUIRE(data.chain.cols() == 30000);
    REQUIRE(data.chain.rows() == 3);

    REQUIRE(data.mean_est.size() == 3);
    REQUIRE(data.median_est.size() == 3);
    REQUIRE(data.upper_95_est.size() == 3);
    REQUIRE(data.lower_95_est.size() == 3);
}

TEST_CASE("norm_post_sim check normal distribution", "[norm_post_sim]") {
    Eigen::Index size     = 3;
    Eigen::VectorXd modes = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(size, size)};
    Sample data = norm_post_sim(modes, cov_matrix);

    for (size_t i = 0; i < size; i++) {
        REQUIRE((data.mean_est[i] > -0.1) & (data.mean_est[i] < 0.1));
        REQUIRE((data.median_est[i] > -0.2) & (data.median_est[i] < 0.2));
        REQUIRE((data.upper_95_est[i] > 1));
        REQUIRE((data.lower_95_est[i] < -1));
    }
}