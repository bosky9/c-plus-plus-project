/**
 * @file test_arima_sunspots_plots.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "arima/arima.hpp"

#include "utilities.hpp" // utils::parse_csv

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test an ARIMA model with sunspot years data (plots included)", "[ARIMA]") {
    utils::DataFrame data = utils::parse_csv("../data/sunspots.csv");

    SECTION("Test prediction length", "[predict]") {
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        model.plot_fit(600, 400);

        delete x;
    }

    SECTION("Test prediction intervals are ordered correctly using BBVI", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("BBVI", opt_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        model.plot_predict(10, 5, false, 600, 400);

        delete x;
    }

    SECTION("Test prediction IS intervals are ordered correctly using BBVI", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("BBVI", opt_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        model.plot_predict_is(10, true, "BBVI", 600, 400);

        delete x;
    }

    SECTION("Test prediction intervals are ordered correctly using MH", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        model.plot_predict(10, 5, false, 600, 400);

        delete x;
    }

    SECTION("Test prediction IS intervals are ordered correctly using MH", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("M-H", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        model.plot_predict_is(10, true, "M-H", 600, 400);

        delete x;
    }

    SECTION("Test prediction intervals are ordered correctly using PML", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("PML", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};

        model.plot_predict(10, 5, false, 600, 400);

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

        model.plot_predict_is(10, true, "PML", 600, 400);

        delete x;
    }

    SECTION("Test prediction intervals are ordered correctly using Laplace", "[predict]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("Laplace", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03,
                             std::nullopt, true)};

        model.plot_predict(10, 5, false, 600, 400);

        delete x;
    }

    SECTION("Test prediction IS intervals are ordered correctly using Laplace", "[predict_is]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
        Results* x{model.fit("Laplace", opt_matrix, 1000, 200, std::nullopt, 12, std::nullopt, true, 1e-03,
                             std::nullopt, true)};

        model.plot_predict_is(10, true, "Laplace", 600, 400);

        delete x;
    }

    SECTION("Test sampling function using BBVI", "[sample]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        model.plot_sample(100, true, 600, 400);

        delete x;
    }

    SECTION("Test sampling function using MH", "[sample]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("M-H", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        model.plot_sample(100, true, 600, 400);

        delete x;
    }

    SECTION("Test PPC value using BBVI", "[ppc]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        model.plot_ppc(1000, utils::mean, "mean", 600, 400);

        delete x;
    }

    SECTION("Test PPC value using MH", "[ppc]") {
        ARIMA model{data, 2, 2};
        std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
        Results* x{
                model.fit("M-H", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

        model.plot_ppc(1000, utils::mean, "mean", 600, 400);

        delete x;
    }
}
