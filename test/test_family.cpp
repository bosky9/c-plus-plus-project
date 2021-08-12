#include "families/family.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Use an empty Family", "[Family]" ) {
    Family f{};
    REQUIRE( f._itransform_name.empty() );
}
