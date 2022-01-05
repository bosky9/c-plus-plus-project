#include "arima/arima.hpp"
#include "utilities.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test sunspot data", "[]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    ARIMA model{data, 10, 10, 0, "sunactivity"};
    std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
    Results* x{
            model.fit("Laplace", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, false)};

    // model.plot_predict_is(10, true, "BBVI", 600, 400);
    // model.plot_predict(5, 20, true, 600, 400);
    //  model.plot_sample(10, true, 600, 400);
    // model.plot_ppc(1000, utils::mean, "mean", 600, 400);

    delete x;
}