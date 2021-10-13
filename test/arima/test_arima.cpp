#include <catch2/catch_test_macros.hpp>

#include "arima/arima.hpp"

#include <random>

TEST_CASE("Tests an ARIMA model with a Normal family", "[ARIMA]") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    SECTION("Tests with no AR or MA terms", "[fit]") {
        ARIMA model{data, 0, 0};
        Results* x{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 2);
        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};
        size_t nan{0};
        for (size_t i{0}; i < lvs.size(); i++) {
            if (!lvs[i].get_value().has_value())
                nan++;
        }
        REQUIRE(nan == 0);
    }
}