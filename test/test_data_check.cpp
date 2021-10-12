#include <catch2/catch_test_macros.hpp>

#include "data_check.hpp"
#include "tsm.hpp"

TEST_CASE("Data check on multiple time series", "[data_check]") {
    std::map<std::string, std::vector<double>> v{{"1", {0.2, 0.3}}, {"2", {0.4, 0.6}}};
    std::vector<double> index{0, 1};
    DataFrame cd = data_check(v, index, "1");
    REQUIRE(cd.data == std::vector<double>{0.2, 0.3});
    REQUIRE(cd.data_name == std::vector<std::string>{"1"});
    REQUIRE(cd.index == std::vector<double>{0, 1});
}

TEST_CASE("Data check a single time series", "[data_check]") {
    std::vector<double> v{0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> index{0, 1, 2, 3, 4};
    DataFrame cd = data_check(v, index);
    REQUIRE(cd.data == std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5});
    REQUIRE(cd.data_name == std::vector<std::string>{"Series"});
    REQUIRE(cd.index == std::vector<double>{0, 1, 2, 3, 4});
}

TEST_CASE("Data check on multiple time series (mv version)", "[mv_data_check]") {
    std::vector<std::vector<double>> v{{0.1, 0.2, 0.3}, {1.2, 1.4, 1.6}};
    CheckedDataMv cd = mv_data_check(v);
    REQUIRE(cd.transformed_data == v);
    REQUIRE(cd.data_name == std::vector<size_t>{1, 2, 3});
    REQUIRE(cd.data_index == std::vector<double>{0, 1, 2});
}