/**
 * @file test_data_check.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "data_check.hpp"

#include "utilities.hpp" // DataFrame

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Data check a single time series", "[data_check]") {
    std::vector<double> v{0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> x(5);
    utils::SingleDataFrame cd = data_check(v, x);
    REQUIRE(cd.data == std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5});
    REQUIRE(cd.data_name == "Series");
    REQUIRE(cd.index == std::vector<double>{0, 1, 2, 3, 4});
}

TEST_CASE("Data check on multiple time series", "[data_check]") {
    utils::DataFrame data_frame;
    data_frame.data      = {{0.2, 0.3}, {0.4, 0.6}};
    data_frame.index     = {0, 1};
    data_frame.data_name = {"1", "2"};
    std::vector<double> x(2);
    utils::SingleDataFrame cd = data_check(data_frame, x, "1");
    REQUIRE(cd.data == std::vector<double>{0.2, 0.3});
    REQUIRE(cd.data_name == "1");
    REQUIRE(cd.index == std::vector<double>{0, 1});
}