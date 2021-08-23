#include "inference/bbvi_routines.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Covariance", "[covariance]") {
    Eigen::VectorXd x1(3); x1 << 0,1,2;
    Eigen::VectorXd y1(3); y1 << 2,1,0;
    REQUIRE(covariance(x1, y1) == -1.0);

    Eigen::VectorXd x2(5); x2 << 1,2,3,4,5;
    Eigen::VectorXd y2(5); y2 << 10,8,6,4,2;
    REQUIRE(covariance(x2, y2) == -5.0);

    Eigen::VectorXd x3(4); x3 << 34, 56, 99, 12;
    Eigen::VectorXd y3(4); y3 << 2, 6, 67, 8;
    REQUIRE(static_cast<int>(covariance(x3, y3)) == 987);

}