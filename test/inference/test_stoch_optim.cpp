/**
 * @file test_stoch_optim.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/stoch_optim.hpp"

#include "Eigen/Core" // Eigen::VectorXd, Eigen::MatrixXd
#include "catch2/catch_test_macros.hpp"

TEST_CASE("Update a RMSProp optimizer", "[update]") {
    RMSProp optimizer{Eigen::Vector2d{0.1, 0.2}, Eigen::Vector2d{0.3, 0.5}, 0.001, 0.04};

    Eigen::VectorXd gradient{Eigen::Vector2d{0.6, 0.7}};
    Eigen::VectorXd parameters{optimizer.update(gradient)};
    REQUIRE(parameters == Eigen::Vector2d{0.1, 0.2});
    REQUIRE(optimizer.get_parameters() == parameters);
}

TEST_CASE("Update an ADAM optimizer", "[update]") {
    ADAM optimizer{Eigen::Vector2d{0.1, 0.2}, Eigen::Vector2d{0.3, 0.5}, 0.001, 0.04, 0.5};

    Eigen::VectorXd gradient{Eigen::Vector2d{0.6, 0.7}};
    Eigen::VectorXd parameters{optimizer.update(gradient)};
    REQUIRE(parameters == Eigen::Vector2d{0.1, 0.2});
}