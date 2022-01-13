/**
 * @file test_utilities.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "utilities.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Get DataFrame from CSV file with 'time' and 'sunactivity' columns", "[parse_csv]") {
    utils::DataFrame df = utils::parse_csv("../data/sunspots.csv");
    if (df.data.empty())
        assert(df.index.empty());
    else
        for (size_t i{0}; i < df.data.size(); ++i)
            assert(df.index.size() == df.data.at(i).size());
    assert(df.data_name.size() == df.data.size());
}