#include <catch2/catch_test_macros.hpp>

#include "latent_variables.hpp"
#include "families/normal.hpp"
#include <type_traits>

TEST_CASE( "LatentVariable creation", "[LatentVariable]" ) {
    LatentVariable lv{"Constant", Normal{0,3}, Normal{0,3}};
}

TEST_CASE( "LatentVariables creation", "[LatentVariables]" ) {
    LatentVariables lvs{"ARIMA"};
}