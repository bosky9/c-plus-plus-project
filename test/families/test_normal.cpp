#include <catch2/catch_test_macros.hpp>

#include "families/family.hpp"
#include "families/normal.hpp"

TEST_CASE("Approximating model", "[approximating_model, approximating_model_reg]") {
    Eigen::VectorXd v = static_cast<Eigen::VectorXd>(Eigen::Vector2d{1, 2});
    Eigen::MatrixXd M = Eigen::MatrixXd::Identity(2, 2);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> H_mu = Normal::approximating_model(v, M, M, M, M, 3, v);
    REQUIRE(H_mu.first == Eigen::MatrixXd::Constant(v.size(), v.size(), 3));
    REQUIRE(H_mu.second == Eigen::MatrixXd::Zero(v.size(), v.size()));

    H_mu = Normal::approximating_model_reg(v, M, M, M, M, 3, v, v, 1);
    REQUIRE(H_mu.first == Eigen::MatrixXd::Constant(v.size(), v.size(), 3));
    REQUIRE(H_mu.second == Eigen::MatrixXd::Zero(v.size(), v.size()));
}

TEST_CASE("Build latent variables", "[build_latent_variables]") {
    Normal normal{};
    std::vector<lv_to_build> result = normal.build_latent_variables();
    REQUIRE(std::get<0>(result.front()) == "Normal scale");
    REQUIRE(std::get<1>(result.front())->get_transform_name() == "exp");
    REQUIRE(*reinterpret_cast<Normal*>(std::get<2>(result.front())) == Normal{0.0, 3.0});
    REQUIRE(std::get<3>(result.front()) == 0.0);
}

TEST_CASE("Draw variable", "[draw_variable, draw_variable_local]") {
    Normal normal{};
    Eigen::VectorXd variable = normal.draw_variable(0.0, 1.0, 4.0, 2.5, 5);
    variable                 = normal.draw_variable_local(2);
}

TEST_CASE("Log of the PDF", "[logpdf]") {
    Normal normal{2.0, 4.0};
    REQUIRE(normal.logpdf(2.0) == -1.3862943611198906);
}

TEST_CASE("Markov blanket", "[markov_blanket") {
    Eigen::VectorXd v      = Eigen::VectorXd::Ones(3);
    Eigen::VectorXd result = Normal::markov_blanket(v, v, 1.0, 3.0, 2.0);
    REQUIRE((result - Eigen::Vector3d{-0.91893853, -0.91893853, -0.91893853}).norm() < 0.0000001);
}

TEST_CASE("Setup", "[setup]") {
    Normal normal{};
    FamilyAttributes attributes = normal.setup();
    REQUIRE(attributes.name == "Normal");
}

TEST_CASE("Negative Log Likelihood", "[neg_loglikelihood]") {
    Eigen::VectorXd v = Eigen::VectorXd::Ones(3);
    REQUIRE(Normal::neg_loglikelihood(v, v, 1.0, 3.0, 2.0) == -2.756815599614018);
}

TEST_CASE("PDF", "[pdf]") {
    Normal normal{};
    REQUIRE(normal.pdf(1.0) == 0.6065306597126334);
}

TEST_CASE("Change parameters", "[vi_change_param, vi_return_param]") {
    Normal normal{};
    normal.vi_change_param(0, 2.0);
    normal.vi_change_param(1, 5.0);
    REQUIRE(normal.vi_return_param(0) == 2.0);
    REQUIRE(normal.vi_return_param(1) == 5.0);
}

TEST_CASE("Compute score", "[vi_score]") {
    Normal normal{};
    REQUIRE(normal.vi_score(1.0, 0) == 1);
    REQUIRE(normal.vi_score(1.0, 1) == 0);
    REQUIRE(normal.vi_score(static_cast<Eigen::VectorXd>(Eigen::Vector2d{1.0, 1.0}), 0) ==
            static_cast<Eigen::VectorXd>(Eigen::Vector2d{1.0, 1.0}));
    REQUIRE(normal.vi_score(static_cast<Eigen::VectorXd>(Eigen::Vector2d{1.0, 1.0}), 1) ==
            static_cast<Eigen::VectorXd>(Eigen::Vector2d{0.0, 0.0}));
}