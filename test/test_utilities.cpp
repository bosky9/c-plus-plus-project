#include <catch2/catch_test_macros.hpp>

#include "utilities.hpp"

TEST_CASE("Get DataFrame from CSV file with 'time' and 'value' columns", "[parse_csv]") {
    DataFrame df = parse_csv("../sunspot.year.csv");
    if (df.data.empty())
        assert(df.index.empty());
    else
        for (size_t i{0}; i < df.data.size(); ++i)
            assert(df.index.size() == df.data.at(i).size());
    assert(df.data_name.size() == df.data.size());
}