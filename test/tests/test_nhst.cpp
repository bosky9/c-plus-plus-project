/**
 * @file test_nhst.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "tests/nhst.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Find p value", "[find_p_value]") {
    REQUIRE(round(find_p_value(1) * 1000) / 1000 == 0.317);
    REQUIRE(round(find_p_value(-2) * 1000) / 1000 == 0.046);
}
