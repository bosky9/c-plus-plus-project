#include "utilities.hpp"
#include "arima/arima.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test sunspot data", "[]") {
    utils::DataFrame df{utils::parse_csv("../data/sunspot.year.csv")};
    if (df.data.empty())
        assert(df.index.empty());
    else
        for (size_t i{0}; i < df.data.size(); ++i)
            assert(df.index.size() == df.data.at(i).size());
    assert(df.data_name.size() == df.data.size());

    ARIMA model{df, 2, 2};

    std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
    Results* x{model.fit("BBVI", opt_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt,
                         true)};
    x->summary(false);

    //utils::DataFrame predictions{model.predict_is(10, true, "MLE", true)};
    model.plot_predict_is(10, true, "MLE", 600, 400);
    //model.plot_predict(5, 1, true, 600, 400); // Controllare valori per past_values

    delete x;
}