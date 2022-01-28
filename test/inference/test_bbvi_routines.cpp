/**
 * @file test_bbvi_routines.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "inference/bbvi_routines.hpp"

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include <catch2/catch_test_macros.hpp>

#include <optional> // std::optional

TEST_CASE("Covariance", "[covariance]") {
    Eigen::Vector3d x1{0, 1, 2};
    Eigen::Vector3d y1{2, 1, 0};
    REQUIRE(bbvi_routines::covariance(x1, y1) == -1.0);

    Eigen::VectorXd x2(5);
    x2 << 1, 2, 3, 4, 5;
    Eigen::VectorXd y2(5);
    y2 << 10, 8, 6, 4, 2;
    REQUIRE(bbvi_routines::covariance(x2, y2) == -5.0);
}

TEST_CASE("Alpha recursion", "[alpha_recursion]") {
    Eigen::VectorXd alpha0{Eigen::Vector2d::Zero()};
    Eigen::MatrixXd m1(2, 3);
    m1 << 3, 5, 7, 4, 7, 9;
    Eigen::MatrixXd m2(2, 3);
    m2 << 12, 11, 10, -3, -8, -9;
    bbvi_routines::alpha_recursion(alpha0, m1, m2, 2);

    REQUIRE(alpha0[0] == -2);
    REQUIRE(static_cast<int64_t>(alpha0[1]) == -7);
}

TEST_CASE("Log p posterior", "[log_p_posterior]") {
    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, 2);
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](Eigen::VectorXd v, [[maybe_unused]] std::optional<size_t> n) { return v[0]; };

    REQUIRE(bbvi_routines::log_p_posterior(z, neg_posterior) == Eigen::Vector2d{-1, -0});
}

TEST_CASE("Mini batch log p posterior", "[mb_log_p_posterior]") {
    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, 2);
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](Eigen::VectorXd v, std::optional<size_t> n) { return v[static_cast<Eigen::Index>(n.value())]; };

    REQUIRE(bbvi_routines::mb_log_p_posterior(z, neg_posterior, 1) == Eigen::Vector2d{-0, -1});
}