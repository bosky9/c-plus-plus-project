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

    // @TODO: ARIMA with 2 AR 2 MA (check if previous two work)

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
        delete x;
    }

    SECTION("Test that the predictions IS are not nans") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame test_df = model.predict_is(5);

        for(auto& vec : test_df.data)
            for(auto& elem : vec)
                REQUIRE(!std::isnan(elem));
        delete x;
    }

    SECTION("Test predictions not having constant values"){
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict(10, false);
        REQUIRE(! (std::adjacent_find( predictions.data.begin(), predictions.data.end(), std::not_equal_to<>() )
                  == predictions.data.end() ) );
        delete x;
    }

    SECTION("Test predictions not having constant values"){
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict_is(10, false);
        REQUIRE(! (std::adjacent_find( predictions.data.begin(), predictions.data.end(), std::not_equal_to<>() )
        == predictions.data.end() ) );
        delete x;
    }

    SECTION("Tests prediction intervals are ordered correctly"){
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict(10, false);

        // forecasted_values, prediction_01, prediction_05, prediction_95, prediction_99
        for(int i = 4; i > 1; i--) {
            double sup_min_element = *min_element(std::begin(predictions.data.at(i)), std::end(predictions.data.at(i)));
            double inf_max_element = *min_element(std::begin(predictions.data.at(i-1)), std::end(predictions.data.at(i-1)));
            REQUIRE(sup_min_element >= inf_max_element);
        }

        double sup_min_element = *min_element(std::begin(predictions.data.at(3)), std::end(predictions.data.at(3)));
        double inf_max_element = *min_element(std::begin(predictions.data.at(0)), std::end(predictions.data.at(0)));
        REQUIRE(sup_min_element >= inf_max_element);

        delete x;
    }

    //@TODO: se giusto, fai tutti gli altri (fino a riga 161)

    SECTION("Test sampling function") {
        ARIMA model{data, 2, 2};
        auto op = std::nullopt;
        // ok cosa
        std::optional<Eigen::MatrixXd>& op_matrix = (std::optional<Eigen::MatrixXd>&) std::nullopt;
        Results* x{model.fit("BBVI", false, op_matrix, 100, op, op, op, op, op, op, op, true)};
        Eigen::MatrixXd sample = model.sample(100);

        REQUIRE(sample.rows() == 100);
        REQUIRE(sample.cols() == data.size()-2);

        delete x;
    }

    SECTION("Test ppc value"){
        ARIMA model{data, 2, 2};
        auto op = std::nullopt;
        std::optional<Eigen::MatrixXd>& op_matrix = (std::optional<Eigen::MatrixXd>&) std::nullopt;
        Results* x{model.fit("BBVI", false, op_matrix, 100, op, op, op, op, op, op, op, true)};
        double p_value = model.ppc();

        REQUIRE(p_value >= 0.0);
        REQUIRE(p_value <= 1.0);

        delete x;
    }
}