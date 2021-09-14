#include <catch2/catch_test_macros.hpp>

#include "families/normal.hpp"
#include "latent_variables.hpp"

TEST_CASE("LatentVariable creation", "[LatentVariable]") {
    LatentVariable lv{"Constant", Normal{0, 3}, Normal{0, 3}};
}

TEST_CASE("Plot latent variable", "[plot_z]") {
    LatentVariable lv{"Constant", Normal{}, Normal{}};
    lv.set_sample({1, 2, 3});
    lv.plot_z(600, 400);
}

TEST_CASE("LatentVariables creation", "[LatentVariables]") {
    LatentVariables lvs{"ARIMA"};
}

TEST_CASE("Plot latent variables", "[plot_z]") {
    LatentVariables lvs{"ARIMA"};
    lvs.create("Constant", std::vector<size_t>{2, 3}, Normal{0, 1}, Normal{1, 2});
    lvs.set_z_values(std::vector<double>{1, 2, 3, 4, 5, 6}, "BBVI", std::vector<double>{2, 4, 6, 8, 10, 12});
    lvs.plot_z(std::vector<size_t>{0, 1, 2, 3}, 600, 400);
}

TEST_CASE("Trace plot", "[trace_plot]") {
    LatentVariables lvs{"ARIMA"};
    lvs.create("Constant", std::vector<size_t>{1, 2}, Normal{0, 1}, Normal{1, 2});
    std::vector<std::vector<double>> samples{{1, 2}, {3, 4}};
    lvs.set_z_values(std::vector<double>{1, 2}, "BBVI", std::vector<double>{2, 4}, samples);
    // lvs.trace_plot();
}