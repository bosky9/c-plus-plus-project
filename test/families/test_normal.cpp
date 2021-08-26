#include <catch2/catch_test_macros.hpp>

#include "families/normal.hpp"

TEST_CASE("Build latent variables", "[build_latent_variables]") {
    Normal normal{2.0, 4.0, "exp"};
    std::list<Normal::lv_to_build> result = normal.build_latent_variables();
    REQUIRE(result.name == "Normal score");
    REQUIRE(result.flat == "exp");
    REQUIRE(result.n == new Normal{0.0, 3.0});
    REQUIRE(result.zero == 0.0);
}

TEST_CASE("Compute score", "[vi_score]") {
    Normal normal{};
    REQUIRE(normal.vi_score(1, 0) == 1);
    REQUIRE(normal.vi_score(1, 1) == 0);
    REQUIRE(normal.vi_score(Eigen::VectorXd{1, 1}, 0) == Eigen::VectorXd{1, 1});
    REQUIRE(normal.vi_score(Eigen::VectorXd{1, 1}, 1) == Eigen::VectorXd{0, 0});
}