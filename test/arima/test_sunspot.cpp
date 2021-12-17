#include "utilities.hpp"
#include "arima/arima.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test sunspot data", "[]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspot.year.csv")};

    ARIMA model{data, 2, 2};
    std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
    Results* x{
            model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, true)};

    /*
    Eigen::MatrixXd sample = model.sample(100);
    REQUIRE(sample.rows() == 100);
    REQUIRE(static_cast<size_t>(sample.cols()) == data.index.size() - 2);
    model.plot_sample(100, true, 600, 400);
     */

    double p_value = model.ppc();
    REQUIRE(p_value >= 0.0);
    REQUIRE(p_value <= 1.0);
    model.plot_ppc(1000, utils::mean, "mean", 600, 400);

    delete x;
}