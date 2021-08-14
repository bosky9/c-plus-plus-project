#include <catch2/catch_test_macros.hpp>

#include "data_check.hpp"

TEST_CASE("Data check on multiple time series", "[data_check]") {
    std::vector<std::vector<double>> v{{0.1, 0.2, 0.3, 0.4, 0.5}, {1.2, 1.4, 1.6}};
    CheckedData* cd = data_check(v, 1);
    REQUIRE(*cd->transformed_data == std::vector<double>{1.2, 1.4, 1.6});
    REQUIRE(*cd->data_name == std::string("Series"));
    REQUIRE(*cd->data_index == std::vector<size_t>{0, 1, 2});

    cd = data_check(v, NULL);
    REQUIRE(*cd->transformed_data == std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5});
    REQUIRE(*cd->data_name == std::string("Series"));
    REQUIRE(*cd->data_index == std::vector<size_t>{0, 1, 2, 3, 4});
}

TEST_CASE("Data check a single time series", "[data_check]") {
    std::vector<double> v{0.1, 0.2, 0.3, 0.4, 0.5};
    CheckedData* cd = data_check(v);
    REQUIRE(*cd->transformed_data == std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5});
    REQUIRE(*cd->data_name == std::string("Series"));
    REQUIRE(*cd->data_index == std::vector<size_t>{0, 1, 2, 3, 4});
}

TEST_CASE("Data check on multiple time series (mv version)", "[mv_data_check]") {
    std::vector<std::vector<double>> v{{0.1, 0.2, 0.3}, {1.2, 1.4, 1.6}};
    CheckedDataMv* cd = mv_data_check(v);
    REQUIRE(*cd->transformed_data == v);
    REQUIRE(*cd->data_name == std::vector<size_t>{1, 2, 3});
    REQUIRE(*cd->data_index == std::vector<size_t>{0, 1, 2});
}