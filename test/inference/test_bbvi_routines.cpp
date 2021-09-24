#include "inference/bbvi_routines.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Covariance", "[covariance]") {
    Eigen::VectorXd x1(3);
    x1 << 0, 1, 2;
    Eigen::VectorXd y1(3);
    y1 << 2, 1, 0;
    REQUIRE(covariance(x1, y1) == -1.0);

    Eigen::VectorXd x2(5);
    x2 << 1, 2, 3, 4, 5;
    Eigen::VectorXd y2(5);
    y2 << 10, 8, 6, 4, 2;
    REQUIRE(covariance(x2, y2) == -5.0);

    Eigen::VectorXd x3(4);
    x3 << 34, 56, 99, 12;
    Eigen::VectorXd y3(4);
    y3 << 2, 6, 67, 8;
    REQUIRE(static_cast<int>(covariance(x3, y3)) == 987);
}

TEST_CASE("Alpha recursion", "[alpha_recursion]") {
    Eigen::VectorXd alpha0(2);
    alpha0 << 0, 0;
    Eigen::MatrixXd m1(2, 3);
    m1 << 3, 5, 7, 4, 7, 9;
    Eigen::MatrixXd m2(2, 3);
    m2 << 12, 11, 10, -3, -8, -9;
    size_t param_no = 2;
    alpha_recursion(alpha0, m1, m2, param_no);
    REQUIRE(alpha0[0] == -2);
    REQUIRE(static_cast<int>(alpha0[1]) == -7);
}

TEST_CASE("Log p posterior", "[log_p_posterior]") {
    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, 2);
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](Eigen::VectorXd v, std::optional<size_t> n) { return v[0]; };

    REQUIRE(log_p_posterior(z, neg_posterior) == Eigen::Vector2d{-1, -0});
}

TEST_CASE("Mini batch log p posterior", "[mb_log_p_posterior]") {
    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, 2);
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](Eigen::VectorXd v, std::optional<size_t> n) { return v[n.value()]; };

    REQUIRE(mb_log_p_posterior(z, neg_posterior, 1) == Eigen::Vector2d{-0, -1});
}