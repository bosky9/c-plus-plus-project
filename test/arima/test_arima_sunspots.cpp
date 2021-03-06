/**
 * @file test_arima_sunspots.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "arima/arima.hpp"

#include "utilities.hpp" // utils::parse_csv

#include <catch2/catch_test_macros.hpp>

namespace catch_utilities {
inline void check_intervals_order(std::vector<std::vector<double>> predictions) {
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

TEST_CASE("Test an ARIMA model with sunspot years data", "[ARIMA]") {
    utils::DataFrame data = utils::parse_csv("../data/sunspots.csv");

    /**
     * @brief Tests on ARIMA model with 1 AR and 1 MA term that the latent variable list length is correct and that
     * the estimated latent variables are not nan
     */
    SECTION("Test an ARIMA model with 1 AR and 1 MA term", "[fit]") {
        ARIMA model{data, 1, 1, 0, "", Normal(0, 3)};
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
        // model.plot_fit(600, 400);

        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};
        int64_t nan{std::count_if(lvs.begin(), lvs.end(),
                                  [](const LatentVariable& lv) { return !lv.get_value().has_value(); })};
        REQUIRE(nan == 0);

        delete x;
    }

    /**
     * @brief Tests on ARIMA model that the summary given by fit results are correct.
     */
    SECTION("Test summary of an ARIMA model fit results", "[fit, summary]") {
        ARIMA model{data, 1, 1, 1};
        Results* x{model.fit()};
        x->summary(true);

        delete x;
    }

    /**
     * @brief Tests that the prediction utils::DataFrame length is equal to the number of steps h
     */
    SECTION("Test prediction length", "[predict]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        // model.plot_fit(600, 400);

        REQUIRE(model.predict(5).data.at(0).size() == 5);

        delete x;
    }

    /**
     * @brief Tests that the in-sample prediction utils::DataFrame length is equal to the number of steps h
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
        utils::DataFrame predictions = model.predict(5);

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
        utils::DataFrame predictions = model.predict_is(5);

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
        utils::DataFrame predictions = model.predict(10, false);
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
        utils::DataFrame predictions = model.predict_is(10, true, "MLE", false);
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

        utils::DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly
     */
    SECTION("Test prediction IS intervals are ordered correctly", "[predict_is]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        utils::DataFrame predictions = model.predict_is(10, true, "MLE", true);
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

        utils::DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict(10, 5, false, 600, 400);

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

        utils::DataFrame predictions = model.predict_is(10, true, "BBVI", true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict_is(10, true, "BBVI", 600, 400);

        delete x;
    }

    /**
     * @brief Tests that prediction intervals are ordered correctly using Metropolis-Hastings method
     */

    SECTION("Test prediction intervals are ordered correctly using MH", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        utils::DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict(10, 5, false, 600, 400);

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

        utils::DataFrame predictions = model.predict_is(10, true, "M-H", true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict_is(10, true, "M-H", 600, 400);

        delete x;
    }

    SECTION("Test prediction intervals are ordered correctly using PML", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("PML", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        utils::DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict(10, 5, false, 600, 400);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly using Metropolis-Hastings method
     */
    SECTION("Test prediction IS intervals are ordered correctly using PML", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("PML", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        utils::DataFrame predictions = model.predict_is(10, true, "PML", true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict_is(10, true, "PML", 600, 400);

        delete x;
    }

    SECTION("Test prediction intervals are ordered correctly using Laplace", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("Laplace", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03,
                             std::nullopt, true)};

        utils::DataFrame predictions = model.predict(10, true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict(10, 5, false, 600, 400);

        delete x;
    }

    /**
     * @brief Tests that in-sample prediction intervals are ordered correctly using Metropolis-Hastings method
     */
    SECTION("Test prediction IS intervals are ordered correctly using Laplace", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("Laplace", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03,
                             std::nullopt, true)};

        utils::DataFrame predictions = model.predict_is(10, true, "Laplace", true);
        catch_utilities::check_intervals_order(predictions.data);
        // model.plot_predict_is(10, true, "Laplace", 600, 400);

        delete x;
    }

    /**
     * @brief Tests sampling function
     */
    SECTION("Test sampling function using BBVI", "[sample]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        Eigen::MatrixXd sample = model.sample(100);
        REQUIRE(sample.rows() == 100);
        REQUIRE(static_cast<size_t>(sample.cols()) == data.index.size() - 2);
        // model.plot_sample(100, true, 600, 400);

        delete x;
    }

    SECTION("Test sampling function using MH", "[sample]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("M-H", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        Eigen::MatrixXd sample = model.sample(100);
        REQUIRE(sample.rows() == 100);
        REQUIRE(static_cast<size_t>(sample.cols()) == data.index.size() - 2);
        // model.plot_sample(100, true, 600, 400);

        delete x;
    }

    /**
     * @brief Tests PPC value
     */
    SECTION("Test PPC value using BBVI", "[ppc]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        double p_value = model.ppc();
        REQUIRE(p_value >= 0.0);
        REQUIRE(p_value <= 1.0);
        // model.plot_ppc(1000, utils::mean, "mean", 600, 400);

        delete x;
    }

    SECTION("Test PPC value using MH", "[ppc]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("M-H", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        double p_value = model.ppc();
        REQUIRE(p_value >= 0.0);
        REQUIRE(p_value <= 1.0);
        // model.plot_ppc(1000, utils::mean, "mean", 600, 400);

        delete x;
    }
}
