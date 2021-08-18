#include <catch2/catch_test_macros.hpp>

#include "inference/norm_post_sim.hpp"

TEST_CASE("norm_post_sim", "[norm_post_sim]") {
    Eigen::VectorXd modes = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(2, 2)};
    NormPostSimData data = norm_post_sim(modes, cov_matrix);
    //REQUIRE(data.chain == Eigen::Matrix<double, Eigen::Dynamic, NSIMS>::Zero(2, 2));
}