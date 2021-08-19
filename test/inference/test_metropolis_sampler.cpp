#include <catch2/catch_test_macros.hpp>

#include "inference/metropolis_sampler.hpp"

TEST_CASE("metropolis_sampler test", "[metropolis_sampler]") {
    Eigen::MatrixXd phi{Eigen::MatrixXd::Identity(3, 3)};
    std::function<double(Eigen::VectorXd)> posterior = [](const Eigen::VectorXd v) { return 0; };
    Eigen::VectorXd a_rate{Eigen::VectorXd::Zero(3)};
    Eigen::MatrixXd rnums{Eigen::MatrixXd::Identity(3, 3)};
    Eigen::VectorXd crit{Eigen::VectorXd::Zero(3)};
    metropolis_sampler(3, phi, posterior, a_rate, rnums, crit);

    Eigen::MatrixXd phi2{Eigen::MatrixXd::Identity(3, 3)};
    phi2.row(1) = Eigen::Vector3d(1, 1, 0);
    phi2.row(2) = Eigen::Vector3d(1, 1, 1);
    REQUIRE((phi - phi2).norm() == 0);
    REQUIRE(a_rate(1) == 1);
}