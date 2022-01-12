/**
 * @file test_latent_variables.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "latent_variables.hpp"

#include "families/flat.hpp"   // Flat
#include "families/normal.hpp" // Normal

#include <catch2/catch_test_macros.hpp>

TEST_CASE("LatentVariable creation", "[LatentVariable]") {
    Normal prior{0, 3};
    Normal q{0, 3};
    std::string name = "Constant";
    LatentVariable lv{name, prior, q};

    SECTION("Get methods") {
        REQUIRE(lv.get_name() == "Constant");
        REQUIRE(lv.get_start() == 0.0);
        REQUIRE(lv.get_prior()->get_name() == "Normal");
        REQUIRE(lv.get_prior()->vi_return_param(0) == 0.0);
        REQUIRE(lv.get_q_clone()->get_name() == "Normal");
        REQUIRE(lv.get_q_clone()->vi_return_param(0) == 0.0);
        REQUIRE(lv.get_method().empty());
        REQUIRE(!lv.get_sample());
        REQUIRE(!lv.get_value());
        REQUIRE(!lv.get_std());
    }

    SECTION("Set methods") {
        lv.set_start(2.0);
        REQUIRE(lv.get_start() == 2.0);
        lv.set_method("AR");
        REQUIRE(lv.get_method() == "AR");
        Flat prior1{};
        lv.set_prior(prior1);
        REQUIRE(lv.get_prior()->get_name() == "Flat");
        Eigen::Vector3d s{2.0, 5.0, 4.0};
        lv.set_sample(s);
        REQUIRE(lv.get_sample().value() == s);
        lv.set_value(3.0);
        REQUIRE(lv.get_value().value() == 3.0);
        lv.set_std(2.0);
        REQUIRE(lv.get_std().value() == 2.0);
    }
}

TEST_CASE("Plot latent variable with sample", "[plot_z]") {
    Normal n1{0, 0.5};
    Normal n2{0.3};
    LatentVariable lv{"AR(1)", n1, n2};
    Eigen::VectorXd sample{{1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, -1, 3}};
    lv.set_sample(sample);
    lv.plot_z();
}

TEST_CASE("Plot latent variable with value and std", "[plot_z]") {
    Normal n1{0, 0.5};
    Normal n2{0.3};
    LatentVariable lv{"AR(1)", n1, n2};
    lv.set_value(3);
    lv.set_std(0.5);
    lv.plot_z();
}

TEST_CASE("Plot latent variable with value, std and prior transform", "[plot_z]") {
    Normal n1{0, 0.5, "exp"};
    Normal n2{0.3};
    LatentVariable lv{"AR(1)", n1, n2};
    lv.set_value(3);
    lv.set_std(0.5);
    lv.plot_z();
}

TEST_CASE("LatentVariables creation", "[LatentVariables]") {
    LatentVariables lvs{"ARIMA"};
}

TEST_CASE("Plot latent variables", "[plot_z]") {
    LatentVariables lvs{"ARIMA"};
    Normal prior{0, 1};
    Normal q{1, 2};
    lvs.create("Constant", std::vector<size_t>{2, 3}, q, prior);
    lvs.set_z_values(Eigen::Matrix<double, 6, 1>{1, 2, 3, 4, 5, 6}, "BBVI",
                     Eigen::Matrix<double, 6, 1>{2, 4, 6, 8, 10, 12});
    lvs.plot_z(std::vector<size_t>{0, 1, 2, 3}, 600, 400);
}

TEST_CASE("Trace plot", "[trace_plot]") {
    Normal prior{0, 1};
    Normal q{1, 2};
    LatentVariables lvs{"ARIMA"};
    lvs.create("Constant", std::vector<size_t>{1, 2}, q, prior);
    Eigen::MatrixXd samples(2, 2);
    samples << 1, 2, 3, 4;
    lvs.set_z_values(Eigen::Vector2d{1, 2}, "BBVI", Eigen::Vector2d{2, 4}, samples);
    lvs.trace_plot(600, 400);
}