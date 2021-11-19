#include <catch2/catch_test_macros.hpp>
#include <random>

#include "arima/arima.hpp"

namespace catch_utilities {
void check_intervals_order(std::vector<std::vector<double>> predictions) {
    REQUIRE(predictions.at(0).size() == predictions.at(1).size());
    REQUIRE(predictions.at(1).size() == predictions.at(2).size());
    REQUIRE(predictions.at(2).size() == predictions.at(3).size());
    REQUIRE(predictions.at(3).size() == predictions.at(4).size());

    // 99% Prediction Interval > 95% Prediction Interval
    for (size_t i{0}; i < predictions.at(4).size(); ++i)
        REQUIRE(predictions.at(4).at(i) >= predictions.at(3).at(i));

    // 95% Prediction Interval > 5% Prediction Interval
    for (size_t i{0}; i < predictions.at(3).size(); ++i)
        REQUIRE(predictions.at(3).at(i) >= predictions.at(2).at(i));

    // 5% Prediction Interval > 1% Prediction Interval
    for (size_t i{0}; i < predictions.at(2).size(); ++i)
        REQUIRE(predictions.at(2).at(i) >= predictions.at(1).at(i));
}
} // namespace catch_utilities

TEST_CASE("Test an ARIMA model with a Normal family", "[ARIMA]") {
    // std::random_device rnd;
    std::default_random_engine generator{};
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; ++i)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    /**
     * @brief Tests that prediction intervals are ordered correctly using Metropolis-Hastings method
     */

    SECTION("Test prediction intervals are ordered correctly using MH", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly using Metropolis-Hastings method
     */
    SECTION("Test prediction IS intervals are ordered correctly using MH", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        DataFrame predictions = model.predict_is(10, true, "MLE", true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests sampling function
     */
    SECTION("Test sampling function", "[sample]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        Eigen::MatrixXd sample = model.sample(100);
        REQUIRE(sample.rows() == 100);
        REQUIRE(static_cast<size_t>(sample.cols()) == data.size() - 2);

        delete x;
    }

    /**
     * @brief Tests PPC value
     */
    SECTION("Test PPC value", "[ppc]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        double p_value = model.ppc();
        REQUIRE(p_value >= 0.0);
        REQUIRE(p_value <= 1.0);

        delete x;
    }
}


TEST_CASE("Test an ARIMA model with a Normal family, 2", "[ARIMA]") {
    // std::random_device rnd;
    std::default_random_engine generator{};
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; ++i)
        data[i] = 0.9 * data[i - 1] + distribution(generator);
    /**
     * @brief Tests on ARIMA model with 1 AR and 1 MA term that the latent variable list length is correct and that
     * the estimated latent variables are not nan
     */
    SECTION("Test an ARIMA model with 1 AR and 1 MA term", "[fit]") {
        ARIMA model{data, 1, 1, 0, Normal(0, 3)};
        Results* x{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 4);

        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};
        int64_t nan{std::count_if(lvs.begin(), lvs.end(),
                                  [](const LatentVariable& lv) { return !lv.get_value().has_value(); })};
        REQUIRE(nan == 0);

        delete x;
    }

    /**
     * @brief Tests on ARIMA model with 1 AR and 1 MA term, integrated once, that the latent variable list length is
     * correct and that the estimated latent variables are not nan
     */
    SECTION("Test an ARIMA model with 1 AR and 1 MA term", "[fit]") {
        ARIMA model{data, 1, 1, 1};
        Results* x{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 4);

        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};
        int64_t nan{std::count_if(lvs.begin(), lvs.end(),
                                  [](const LatentVariable& lv) { return !lv.get_value().has_value(); })};
        REQUIRE(nan == 0);

        delete x;
    }

    /**
     * @brief Tests that the prediction dataframe length is equal to the number of steps h
     */
    SECTION("Test prediction length", "[predict]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        REQUIRE(model.predict(5).data.at(0).size() == 5);

        delete x;
    }

    /**
     * @brief Tests that the in-sample prediction dataframe length is equal to the number of steps h
     */
    SECTION("Test prediction IS length", "[predict_is]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        REQUIRE(model.predict_is(5).data.at(0).size() == 5);

        delete x;
    }

    /**
     * @brief Tests that the predictions are not nans
     */
    SECTION("Test predictions are not nans", "[predict]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict(5);

        for (auto& v : predictions.data)
            REQUIRE(std::count_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }) == 0);

        delete x;
    }

    /**
     * @brief Tests that the in-sample predictions are not nans
     */
    SECTION("Test predictions IS are not nans", "[predict_is]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict_is(5);

        for (auto& v : predictions.data)
            REQUIRE(std::count_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }) == 0);

        delete x;
    }

    /**
     * @brief We should not really have predictions that are constant...
     * This captures bugs with the predict function not iterating forward.
     */
    SECTION("Test predictions not having constant values", "[predict]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict(10, false);
        REQUIRE(std::adjacent_find(predictions.data.at(0).begin(), predictions.data.at(0).end(),
                                   std::not_equal_to<>()) != predictions.data.at(0).end());
        delete x;
    }

    /**
     * @brief We should not really have predictions that are constant...
     * This captures bugs with the predict function not iterating forward.
     */
    SECTION("Test predictions IS not having constant values", "[predict_is]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict_is(10, true, "MLE", false);
        REQUIRE(std::adjacent_find(predictions.data.at(0).begin(), predictions.data.at(0).end(),
                                   std::not_equal_to<>()) != predictions.data.at(0).end());
        delete x;
    }

    /**
     * @brief Tests that prediction intervals are ordered correctly
     */
    SECTION("Test prediction intervals are ordered correctly", "[predict]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly
     */
    SECTION("Test prediction IS intervals are ordered correctly", "[predict_is]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        DataFrame predictions = model.predict_is(10, true, "MLE", true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests that prediction intervals are ordered correctly using BBVI method
     */
    SECTION("Test prediction intervals are ordered correctly using BBVI", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("BBVI", opt_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly using BBVI method
     */
    SECTION("Test prediction IS intervals are ordered correctly using BBVI", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("BBVI", opt_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        DataFrame predictions = model.predict_is(10, true, "MLE", true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    // ---  Up to this point, no memory leaks are detected ---

    /**
     * @brief Tests that prediction intervals are ordered correctly using Metropolis-Hastings method
     */

    SECTION("Test prediction intervals are ordered correctly using MH", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly using Metropolis-Hastings method
     */
    SECTION("Test prediction IS intervals are ordered correctly using MH", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        DataFrame predictions = model.predict_is(10, true, "MLE", true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests sampling function
     */
    SECTION("Test sampling function", "[sample]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        Eigen::MatrixXd sample = model.sample(100);
        REQUIRE(sample.rows() == 100);
        REQUIRE(static_cast<size_t>(sample.cols()) == data.size() - 2);

        delete x;
    }

    /**
     * @brief Tests PPC value
     */
    SECTION("Test PPC value", "[ppc]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        double p_value = model.ppc();
        REQUIRE(p_value >= 0.0);
        REQUIRE(p_value <= 1.0);

        delete x;
    }
}
