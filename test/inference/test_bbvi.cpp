/**
 * @file test_bbvi.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "inference/bbvi.hpp"

#include <catch2/catch_test_macros.hpp>
#include "families/normal.hpp" // Normal
#include "inference/bbvi_routines.hpp"
#include "multivariate_normal.hpp"

#include <iostream>
#include <limits> // std::numeric_limits<double>::epsilon()

TEST_CASE("Test a BBVI object", "[BBVI]") {
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](const Eigen::VectorXd& v, std::optional<size_t> = std::nullopt) { return v[0]; };
    std::vector<std::unique_ptr<Family>> q;
    q.push_back(std::make_unique<Normal>(Normal()));
    q.push_back(std::make_unique<Normal>(Normal(0.6, 0.2)));
    BBVI bbvi = BBVI(neg_posterior, q, 2, "ADAM", 100, 0.01, false, false);

    SECTION("Test equality operator") {
        BBVI bbvi2{bbvi};
        REQUIRE(bbvi2 == bbvi);
        BBVI bbvi3{std::move(bbvi2)};
        REQUIRE(bbvi3 == bbvi);
        BBVI bbvi4 = BBVI(neg_posterior, q, 0);
        bbvi4      = bbvi;
        bbvi4      = bbvi3;
        REQUIRE(bbvi4 == bbvi);
        bbvi4 = std::move(bbvi3);
        REQUIRE(bbvi4 == bbvi);
    }

    SECTION("Change parameters", "[change_parameters, current_parameters]") {
        Eigen::VectorXd params{static_cast<Eigen::VectorXd>(Eigen::Vector4d{3.0, 1.0, 2.0, 1.0})};
        bbvi.change_parameters(params);
        REQUIRE(bbvi.current_parameters() == params);
    }

    SECTION("Compute cv_gradient", "[cv_gradient]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        REQUIRE((bbvi.cv_gradient(z, true) - Eigen::Vector4d{-2.11421958, 1.11421958, -3.42890212, -39.45641535})
                        .norm() < 0.00000001);

        auto result{bbvi.run(false)};
        auto gradient{bbvi.cv_gradient(z, false)};
    }

    SECTION("Draw normal", "[draw_normal]") {
        auto normal{bbvi.draw_normal(true)};
        auto result{bbvi.run(false)};
        normal = bbvi.draw_normal(false);
    }

    SECTION("Draw variables", "[draw_variables]") {
        auto variables{bbvi.draw_variables()};
    }

    SECTION("Get means and scales from q", "[get_means_and_scales_from_q]") {
        auto means_scales = bbvi.get_means_and_scales_from_q();
        REQUIRE(means_scales.first == Eigen::Vector2d{0, 0.6});
        REQUIRE(means_scales.second == Eigen::Vector2d{1, 0.2});
    }

    SECTION("Get means and scales", "[get_means_and_scales]") {
        auto result{bbvi.run(false)};
        auto means_scales{bbvi.get_means_and_scales()};
    }

    SECTION("Compute the gradient of the approximating distributions", "[grad_log_q]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        Eigen::MatrixXd result(4, 2);
        result << 1, 0, 0, -1, -15, 10, 8, 3;
        REQUIRE((bbvi.grad_log_q(z) - result).norm() < 0.1);
    }

    SECTION("Compute the unnormalized log posterior components", "[log_p]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        REQUIRE(bbvi.log_p(z) == Eigen::Vector2d{-1, -0});
    }

    SECTION("Compute the mean-field normal log posterior components", "[normal_log_q]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        REQUIRE((bbvi.normal_log_q(z, true) - Eigen::Vector2d{-5.22843915, -2.22843915}).norm() < 0.00000001);
        auto result{bbvi.run(false)};
        auto normal_log_q_res{bbvi.normal_log_q(z, false)};
    }

    SECTION("Print progress", "[print_progress]") {
        bbvi.print_progress(10, bbvi.current_parameters());
    }

    SECTION("Get ELBO", "[get_elbo]") {
        auto result{bbvi.run(false)};
        auto elbo{bbvi.get_elbo(bbvi.current_parameters()(Eigen::seq(0, 1)))};
        REQUIRE(elbo >= 0);
    }

    SECTION("Run", "[run, run_with]") {
        auto result{bbvi.run(true)};
    }
}
TEST_CASE("Test a CBBVI object", "[CBBVI]") {
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](const Eigen::VectorXd& v, std::optional<size_t> = std::nullopt) { return v[0]; };
    std::function<Eigen::VectorXd(Eigen::VectorXd)> log_p_blanket = [](const Eigen::VectorXd& v) { return v; };
    std::vector<std::unique_ptr<Family>> q;
    q.push_back(std::make_unique<Normal>(Normal()));
    q.push_back(std::make_unique<Normal>(Normal(0.6, 0.2)));
    CBBVI cbbvi = CBBVI(neg_posterior, log_p_blanket, q, 2, "ADAM", 100, 0.01, false, false);

    SECTION("Test equality operator") {
        CBBVI cbbvi2{cbbvi};
        REQUIRE(cbbvi2 == cbbvi);
        CBBVI cbbvi3{std::move(cbbvi2)};
        REQUIRE(cbbvi3 == cbbvi);
        CBBVI cbbvi4 = CBBVI(neg_posterior, log_p_blanket, q, 0);
        cbbvi4       = cbbvi;
        cbbvi4       = cbbvi3;
        REQUIRE(cbbvi4 == cbbvi);
        cbbvi4 = std::move(cbbvi3);
        REQUIRE(cbbvi4 == cbbvi);
    }

    SECTION("Compute the unnormalized log posterior components", "[log_p]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        REQUIRE(cbbvi.log_p(z) == Eigen::Matrix2d::Identity());
    }

    SECTION("Compute the mean-field normal log posterior components", "[normal_log_q]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        Eigen::MatrixXd normal_log_q_res(2, 2);
        normal_log_q_res << -1.41893853, -3.80950062, -0.91893853, -1.30950062;
        REQUIRE((cbbvi.normal_log_q(z, true) - normal_log_q_res).norm() < 0.00000001);
        auto result{cbbvi.run(false)};
        normal_log_q_res = cbbvi.normal_log_q(z, false);
    }

    SECTION("Compute cv_gradient", "[cv_gradient]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        REQUIRE((cbbvi.cv_gradient(z, true) - Eigen::Vector4d{-1.20946927, 0.45946927, -0.97624845, -33.10225341})
                        .norm() < 0.00000001);

        auto result{cbbvi.run(false)};
        auto gradient{cbbvi.cv_gradient(z, false)};
    }
}

TEST_CASE("Test a BBVIM object", "[BBVIM]") {
    std::function<double(Eigen::VectorXd, std::optional<size_t>)> neg_posterior =
            [](const Eigen::VectorXd& v, std::optional<size_t> = std::nullopt) { return v[0]; };
    std::function<double(Eigen::VectorXd)> full_neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<std::unique_ptr<Family>> q;
    q.push_back(std::make_unique<Normal>(Normal()));
    q.push_back(std::make_unique<Normal>(Normal(0.6, 0.2)));
    BBVIM bbvim = BBVIM(neg_posterior, full_neg_posterior, q, 2, "RMSProp", 100, 0.01, false, false);

    SECTION("Test equality operator") {
        BBVIM bbvim2{bbvim};
        REQUIRE(bbvim2 == bbvim);
        BBVIM bbvim3{std::move(bbvim2)};
        REQUIRE(bbvim3 == bbvim);
        BBVIM bbvim4 = BBVIM(neg_posterior, full_neg_posterior, q, 0);
        bbvim4       = bbvim;
        bbvim4       = bbvim3;
        REQUIRE(bbvim4 == bbvim);
        bbvim4 = std::move(bbvim3);
        REQUIRE(bbvim4 == bbvim);
    }

    SECTION("Compute the unnormalized log posterior components", "[log_p]") {
        Eigen::MatrixXd z{Eigen::MatrixXd::Identity(2, 2)};
        REQUIRE(bbvim.log_p(z) == Eigen::Vector2d{-1, -0});
    }

    SECTION("Get ELBO (for BBVIM)", "[get_elbo]") {
        auto result{bbvim.run(false)};
        auto elbo{bbvim.get_elbo(bbvim.current_parameters()(Eigen::seq(0, 1)))};
        REQUIRE(elbo >= 0);
    }

    SECTION("Print progress", "[print_progress]") {
        bbvim.print_progress(10, bbvim.current_parameters());
    }

    SECTION("Run", "[run, run_with]") {
        auto result{bbvim.run(true)};
    }
}