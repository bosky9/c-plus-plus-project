/**
 * @file test_flat.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "families/flat.hpp"

#include "catch2/catch_test_macros.hpp"

TEST_CASE("Log PDF", "[logpdf]") {
    Flat flat{"logit"};
    REQUIRE(flat.logpdf(1.0) == 0.0);
}