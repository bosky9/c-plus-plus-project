/**
 * @file test_metropolis_hastings.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/metropolis_sampler.hpp"

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include "catch2/catch_test_macros.hpp"

TEST_CASE("Test Metropolis sampler", "[metropolis_sampler]") {
    Eigen::MatrixXd phi{Eigen::MatrixXd::Identity(3, 3)};
    std::function<double(const Eigen::VectorXd&)> posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    Eigen::VectorXd a_rate{Eigen::VectorXd::Constant(3, 0.5)};
    Eigen::MatrixXd rnums{Eigen::MatrixXd::Identity(3, 3)};
    Eigen::VectorXd crit{Eigen::VectorXd::Zero(3)};
    metropolis::metropolis_sampler(3, phi, posterior, a_rate, rnums, crit);

    Eigen::MatrixXd phi_post(3, 3);
    phi_post << 1, 0, 0, 1, 1, 0, 1, 1, 1;
    REQUIRE(phi == phi_post);
    REQUIRE(a_rate == Eigen::Vector3d{0.5, 1, 1});
}