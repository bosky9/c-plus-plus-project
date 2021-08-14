#include "families/family.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Use an empty Family", "[Family]" ) {
    Family f{};
    REQUIRE(f._transform_name.empty());
    REQUIRE(f._itransform_name.empty());
    REQUIRE(f._transform(3.0) == 3.0);
    REQUIRE(f._itransform(3.0) == 3.0);
}

TEST_CASE( "Use a wrong Family", "[Family]" ) {
    Family f{"wrong_name"};
    REQUIRE(f._transform_name == "wrong_name");
    REQUIRE(f._itransform_name.empty());
    REQUIRE(f._transform(3.0) == 3.0);
    REQUIRE(f._itransform(3.0) == 3.0);
}

TEST_CASE( "Use an exp Family", "[Family]" ) {
    Family f{"exp"};
    REQUIRE(f._transform_name == "exp");
    REQUIRE(f._itransform_name == "log");
    REQUIRE(round(f._transform(3.0)*100)/100 == round(exp(3.0)*100)/100);
    REQUIRE(round(f._itransform(3.0)*100)/100 == round(log(3.0)*100)/100);
}

TEST_CASE( "Use a tanh Family", "[Family]" ) {
    Family f{"tanh"};
    REQUIRE(f._transform_name == "tanh");
    REQUIRE(f._itransform_name == "arctanh");
    REQUIRE(round(f._transform(0.5)*100)/100 == round(tanh(0.5)*100)/100);
    REQUIRE(round(f._itransform(0.5)*100)/100 == round(atanh(0.5)*100)/100);
}

TEST_CASE( "Use a logit Family", "[Family]" ) {
    Family f{"logit"};
    REQUIRE(f._transform_name == "logit");
    REQUIRE(f._itransform_name == "ilogit");
    REQUIRE(round(f._transform(0.5)*100)/100 == round((1.0/(1.0+exp(-0.5)))*100)/100);
    REQUIRE(round(f._itransform(0.5)*100)/100 == round((log(0.5)-log(1.0-0.5))*100)/100);
}