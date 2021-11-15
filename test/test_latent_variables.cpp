#include <catch2/catch_test_macros.hpp>

#include "families/flat.hpp"
#include "families/normal.hpp"
#include "latent_variables.hpp"

#include <memory>

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
        REQUIRE(lv.get_q()->get_name() == "Normal");
        REQUIRE(lv.get_q()->vi_return_param(0) == 0.0);
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

TEST_CASE("Plot latent variable", "[plot_z]") {
    Normal prior{};
    Normal q{};
    std::string name = "Constant";
    LatentVariable lv{name, prior, q};
    lv.set_sample(Eigen::Vector3d{1, 2, 3});
    lv.plot_z(600, 400);
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