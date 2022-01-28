/**
 * @file test_family.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "families/family.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Use an empty Family", "[Family]") {
    Family f{};
    REQUIRE(f.get_transform_name().empty());
    REQUIRE(f.get_itransform_name().empty());
    REQUIRE(f.get_transform()(3.0) == 3.0);
    REQUIRE(f.get_itransform()(3.0) == 3.0);
}

TEST_CASE("Use an exp Family", "[Family]") {
    Family f{"exp"};
    REQUIRE(f.get_transform_name() == "exp");
    REQUIRE(f.get_itransform_name() == "log");
    REQUIRE(round(f.get_transform()(3.0) * 100) / 100 == round(exp(3.0) * 100) / 100);
    REQUIRE(round(f.get_itransform()(3.0) * 100) / 100 == round(log(3.0) * 100) / 100);
}

TEST_CASE("Use a tanh Family", "[Family]") {
    Family f{"tanh"};
    REQUIRE(f.get_transform_name() == "tanh");
    REQUIRE(f.get_itransform_name() == "arctanh");
    REQUIRE(round(f.get_transform()(0.5) * 100) / 100 == round(tanh(0.5) * 100) / 100);
    REQUIRE(round(f.get_itransform()(0.5) * 100) / 100 == round(atanh(0.5) * 100) / 100);
}

TEST_CASE("Use a logit Family", "[Family]") {
    Family f{"logit"};
    REQUIRE(f.get_transform_name() == "logit");
    REQUIRE(f.get_itransform_name() == "ilogit");
    REQUIRE(round(f.get_transform()(0.5) * 100) / 100 == round((1.0 / (1.0 + exp(-0.5))) * 100) / 100);
    REQUIRE(round(f.get_itransform()(0.5) * 100) / 100 == round((log(0.5) - log(1.0 - 0.5)) * 100) / 100);
}