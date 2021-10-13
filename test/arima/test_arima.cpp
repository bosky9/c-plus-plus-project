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
            if (!lvs[i].get_value())
                nan++;
        }
        REQUIRE(nan == 0);

        delete x;
    }

    SECTION("Tests an ARIMA model with 1 AR and 1 MA term", "[fit]") {
        ARIMA model{data, 1, 1};
        Results* x{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 4);
        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};

        size_t nan{0};
        for (size_t i{0}; i < lvs.size(); i++) {
            if (!lvs[i].get_value())
                nan++;
        }
        REQUIRE(nan == 0);

        delete x;
    }

    SECTION("Test prediction length", "[fit]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        REQUIRE(model.predict(5).data.at(0).size() == 5);

        delete x;
    }

    SECTION("Test prediction IS length", "[fit]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        REQUIRE(model.predict_is(5).data.at(0).size() == 5);

        delete x;
    }

    SECTION("Test that the predictions are not nans") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame test_df = model.predict(5);

        for(auto& vec : test_df.data)
            for(auto& elem : vec)
                REQUIRE(!std::isnan(elem));

    }

    SECTION("Test that the predictions IS are not nans") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame test_df = model.predict_is(5);

        for(auto& vec : test_df.data)
            for(auto& elem : vec)
                REQUIRE(!std::isnan(elem));

    }


}