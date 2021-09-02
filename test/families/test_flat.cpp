#include <catch2/catch_test_macros.hpp>

#include "families/flat.hpp"

TEST_CASE("Log PDF", "[logpdf]") {
    Flat flat{"logit"};
    REQUIRE(flat.get_covariance_prior() == false);
    REQUIRE(flat.logpdf(1.0) == 0.0);
}