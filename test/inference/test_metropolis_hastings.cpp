#include <catch2/catch_test_macros.hpp>

#include "inference/metropolis_hastings.hpp"

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