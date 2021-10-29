#include <catch2/catch_test_macros.hpp>

#include "families/flat.hpp"
#include "families/normal.hpp"
#include "results.hpp"

/*
TEST_CASE("Create MLEResults object", "[MLEResults]") {
    LatentVariables lvs{"ARIMA"};
    Normal norm{2, 0.5};
    Flat flat{};
    lvs.create("lv", {3}, norm, flat);
    lvs.set_z_values(Eigen::Vector2d(0.4, 0.6), "MLE");
    std::function<double(const Eigen::VectorXd&)> posterior{[](const Eigen::VectorXd& x) { return x[0]; }};
    MLEResults mler({"Differenced Series"}, {}, "Normal ARIMA(1,2,3)", "ARIMA", lvs, Eigen::VectorXd::Constant(1, 0.3),
                    Eigen::VectorXd::Constant(1, 2), {0}, false, posterior, "Metropolis Hastings", false, 3);

    SECTION("Print", "[operator<<]") { // FIXME: operator<< not found
        std::cout << mler << "\n";
    }

    SECTION("Summary", "[summary, summary_without_hessian]") {
        mler.summary(false);
    }

    SECTION("Summary", "[summary, summary_with_hessian]") {
        MLEResults mler_h({"Differenced Series"}, {}, "Normal ARIMA(1,2,3)", "ARIMA", lvs,
                          Eigen::VectorXd::Constant(2, 0.3), Eigen::MatrixXd::Identity(2, 2), {0}, false, posterior,
                          "Metropolis Hastings", false, 3, Eigen::MatrixXd::Identity(2, 2));

        mler_h.summary(false);
        mler_h.summary(true);
    }
}

TEST_CASE("Create BBVIResults object", "[BBVIResults]") {
    LatentVariables lvs{"ARIMA"};
    Normal norm{2, 0.5};
    Flat flat{};
    lvs.create("lv", {3}, norm, flat);
    std::function<double(const Eigen::VectorXd&)> posterior{[](const Eigen::VectorXd& x) { return x[0]; }};
    BBVIResults bbvir({"Differenced Series"}, {}, "Normal ARIMA(1,2,3)", "ARIMA", lvs, Eigen::MatrixXd::Identity(2, 2),
                      {0}, false, posterior, "Metropolis Hastings", false, 3, Eigen::VectorXd::Constant(2, 0.2));

    SECTION("Print", "[operator<<]") { // FIXME: operator<< not found
        std::cout << bbvir << "\n";
    }

    SECTION("Summary", "[summary, summary_without_hessian]") {
        bbvir.summary(false);
    }

    SECTION("Summary", "[summary, summary_with_hessian]") {
        BBVIResults bbvir_h({"Differenced Series"}, {}, "Normal ARIMA(1,2,3)", "ARIMA", lvs,
                            Eigen::MatrixXd::Identity(2, 2), {0}, false, posterior, "Metropolis Hastings", false, 3,
                            Eigen::VectorXd::Constant(2, 0.2), Eigen::MatrixXd::Identity(2, 2));

        bbvir_h.summary(false);
        bbvir_h.summary(true);
    }

    SECTION("Plot elbo", "[plot_elbo]") {
        bbvir.plot_elbo();
    }
}

TEST_CASE("Create LaplaceResults object", "[LaplaceResults]") {
    LatentVariables lvs{"ARIMA"};
    Normal norm{2, 0.5};
    Flat flat{};
    lvs.create("lv", {3}, norm, flat);
    std::function<double(const Eigen::VectorXd&)> posterior{[](const Eigen::VectorXd& x) { return x[0]; }};
    LaplaceResults lr({"Differenced Series"}, {}, "Normal ARIMA(1,2,3)", "ARIMA", lvs, Eigen::MatrixXd::Identity(2, 2),
                      {0}, false, posterior, "Metropolis Hastings", false, 3, Eigen::VectorXd::Constant(2, 0.2));

    SECTION("Print", "[operator<<]") { // FIXME: operator<< not found
        std::cout << lr << "\n";
    }

    SECTION("Summary", "[summary, summary_without_hessian]") {
        lr.summary(false);
    }

    SECTION("Summary", "[summary, summary_with_hessian]") {
        BBVIResults lr_h({"Differenced Series"}, {}, "Normal ARIMA(1,2,3)", "ARIMA", lvs,
                         Eigen::MatrixXd::Identity(2, 2), {0}, false, posterior, "Metropolis Hastings", false, 3,
                         Eigen::VectorXd::Constant(2, 0.2), Eigen::MatrixXd::Identity(2, 2));

        lr_h.summary(false);
        lr_h.summary(true);
    }
}*/
