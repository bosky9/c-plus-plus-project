#include <catch2/catch_test_macros.hpp>

#include "inference/metropolis_hastings.hpp"

#include "inference/metropolis_sampler.hpp"


TEST_CASE("New MetropolisHastings object without cov_matrix", "[MetropolisHastings]") {
    std::function<double(Eigen::VectorXd)> posterior = [](const Eigen::VectorXd& v) { return 0; };
    Eigen::VectorXd initials{Eigen::VectorXd::Zero(3)};
    MetropolisHastings mh = MetropolisHastings(posterior, 1.0, 3, initials);
}

TEST_CASE("New MetropolisHastings object with cov_matrix", "[MetropolisHastings]") {
    std::function<double(Eigen::VectorXd)> posterior = [](const Eigen::VectorXd& v) { return 0; };
    Eigen::VectorXd initials{Eigen::VectorXd::Zero(3)};
    Eigen::MatrixXd cov_matrix{Eigen::MatrixXd::Identity(3, 3)};
    MetropolisHastings mh = MetropolisHastings(posterior, 1.0, 3, initials, cov_matrix, 3, false, true);
}

TEST_CASE("Tune scale", "[tune_scale]") {
    std::function<double(Eigen::VectorXd)> posterior = [](const Eigen::VectorXd& v) { return 0; };
    Eigen::VectorXd initials{Eigen::VectorXd::Zero(3)};
    MetropolisHastings mh = MetropolisHastings(posterior, 1.0, 3, initials);
    REQUIRE(mh.tune_scale(0.6, 1.0) == 1.3);
}

TEST_CASE("metropolis_sampler test", "[metropolis_sampler]") {
    Eigen::MatrixXd phi{Eigen::MatrixXd::Identity(3, 3)};
    std::function<double(Eigen::VectorXd)> posterior = [](const Eigen::VectorXd& v) { return 0; };
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