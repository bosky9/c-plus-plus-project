#include <catch2/catch_test_macros.hpp>

#include "data_check.hpp"
#include "tsm.hpp"

TEST_CASE("Data check a single time series", "[data_check]") {
    std::vector<double> v{0.1, 0.2, 0.3, 0.4, 0.5};
    SingleDataFrame cd = data_check(v);
    REQUIRE(cd.data == std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5});
    REQUIRE(cd.data_name == "Series");
    REQUIRE(cd.index == std::vector<double>{0, 1, 2, 3, 4});
}

TEST_CASE("Data check on multiple time series", "[data_check]") {
    DataFrame data_frame;
    data_frame.data      = {{0.2, 0.3}, {0.4, 0.6}};
    data_frame.index     = {0, 1};
    data_frame.data_name = {"1", "2"};
    SingleDataFrame cd   = data_check(data_frame, "1");
    REQUIRE(cd.data == std::vector<double>{0.2, 0.3});
    REQUIRE(cd.data_name == "1");
    REQUIRE(cd.index == std::vector<double>{0, 1});
}

TEST_CASE("Data check on single time series (mv version)", "[mv_data_check]") {
    std::vector<std::vector<double>> v{{0.1, 0.2, 0.3}, {1.2, 1.4, 1.6}};
    DataFrame cd = mv_data_check(v);
    REQUIRE(cd.data == v);
    REQUIRE(cd.data_name == std::vector<std::string>{"1", "2", "3"});
    REQUIRE(cd.index == std::vector<double>{0, 1, 2});
}

TEST_CASE("Data check on multiple time series (mv version)", "[mv_data_check]") {
    DataFrame data_frame;
    data_frame.data      = {{0.1, 0.2, 0.3}, {1.2, 1.4, 1.6}};
    data_frame.index     = {0, 1, 2};
    data_frame.data_name = {"1", "2", "3"};
    DataFrame cd         = mv_data_check(data_frame);
    REQUIRE(cd.data == data_frame.data);
    REQUIRE(cd.data_name == data_frame.data_name);
    REQUIRE(cd.index == data_frame.index);
}