#include "utilities.hpp"
#include "arima/arima.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test sunspot data", "[parse_csv]") {
    utils::DataFrame df = utils::parse_csv("../data/sunspot.year.csv");
    if (df.data.empty())
        assert(df.index.empty());
    else
        for (size_t i{0}; i < df.data.size(); ++i)
            assert(df.index.size() == df.data.at(i).size());
    assert(df.data_name.size() == df.data.size());

    ARIMA model{df, 4, 4};

    Results* x = model.fit("MLE");
    x->summary(false);

    delete x;
}