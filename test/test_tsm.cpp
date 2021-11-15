#include <catch2/catch_test_macros.hpp>

#include "arima/arima.hpp"
#include "tsm.hpp"

#include <random>

TEST_CASE("Tests an ARIMA model", "[TSM]") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; ++i)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    SECTION("Fit", "[fit]") {
        ARIMA model{data, 1, 2};
        Results* x{model.fit()};
    }
}