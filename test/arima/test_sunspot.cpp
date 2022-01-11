/**
 * @file test_sunspot.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "arima/arima.hpp"

#include "utilities.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test sunspot data", "[]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    ARIMA model{data, 2, 2, 0, "sunactivity"};
    std::optional<Eigen::MatrixXd> op_matrix = std::nullopt;
    Results* x{model.fit("BBVI", op_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt, false)};

    model.plot_fit();
    model.plot_predict_is(10, true, "BBVI");
    model.plot_predict(5, 20, true);
    model.plot_sample(10, true);
    model.plot_ppc(1000, utils::mean, "mean");

    model.get_latent_variables().get_z_list()[0].plot_z();
    model.get_latent_variables().get_z_list()[1].plot_z();
    model.get_latent_variables().get_z_list()[2].plot_z();
    model.get_latent_variables().get_z_list()[3].plot_z();
    // model.plot_z(std::nullopt, 600, 400);
    //   auto y = model.draw_latent_variables();
    //   model.get_latent_variables().trace_plot(1200, 800);

    delete x;
}