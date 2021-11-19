/**
 * @file test_metropolis_hastings.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/metropolis_hastings.hpp"

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include "catch2/catch_test_macros.hpp"
#include "inference/sample.hpp" // Sample

TEST_CASE("Test Metropolis-Hastings", "[MetropolisHastings]") {
    std::function<double(const Eigen::VectorXd&)> posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    Eigen::VectorXd initials{Eigen::Vector3d{0.1, 0.2, 0.3}};
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(3, 3)};
    MetropolisHastings mh = MetropolisHastings(posterior, 1.0, 3, initials, cov_matrix, 3, false, true);

    REQUIRE(mh.tune_scale(0.6, 1.0) == 1.3);

    /*
    SECTION("Sample", "[sample]") {
        Sample sample{mh.sample()};
    }*/
}