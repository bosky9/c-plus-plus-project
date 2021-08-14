#include "covariances.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Double covariances", "[cov]" ) {
    std::vector<double> x{1.0, 2.0, 6.0, 13.0, 10.0};
    REQUIRE(round(cov(x)*100)/100 == 21.04);
    REQUIRE(round(cov(x,1)*100)/100 == 11.66);
    REQUIRE(round(cov(x,2)*100)/100 == -9.44);
    REQUIRE(round(cov(x,3)*100)/100 == -25.74);
}