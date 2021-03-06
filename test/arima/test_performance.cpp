/**
 * @file test_performance.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "arima/arima.hpp"

#include "utilities.hpp" // utils::parse_csv, utils::create_performance_file, utils::save_performance

#include <catch2/catch_test_macros.hpp>
#include <chrono> // std::chrono::steady_clock::now, std::chrono::duration_cast

TEST_CASE("Test performances on sunspot data (no sample and ppc)", "") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};
    size_t ar = 2;
    size_t ma = 2;
    ARIMA model{data, ar, ma, 0, "sunactivity"};
    std::string method{"Laplace"};
    std::string filename{utils::create_performance_file("sunspots", method, ar, ma)};

    SECTION("fit", "[fit]") {
        auto start{std::chrono::steady_clock::now()};
        Results* x{model.fit(method)};
        auto end{std::chrono::steady_clock::now()};
        utils::save_performance(filename, "fit", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }

    SECTION("predict", "[predict]") {
        Results* x{model.fit(method)};
        auto start{std::chrono::steady_clock::now()};
        utils::DataFrame predictions{model.predict()};
        auto end{std::chrono::steady_clock::now()};
        utils::save_performance(filename, "predict", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }

    SECTION("predict_is", "[predict_is]") {
        Results* x{model.fit(method)};
        auto start = std::chrono::steady_clock::now();
        utils::DataFrame predictions{model.predict_is(50, true, method)};
        auto end = std::chrono::steady_clock::now();
        utils::save_performance(filename, "predict_is", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }
}

TEST_CASE("Test performances on sunspot data (full)", "") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};
    ARIMA model{data, 2, 2, 0, "sunactivity"};
    std::string method{"BBVI"};
    std::string filename{utils::create_performance_file("sunspots", method, 2, 2, 0)};

    SECTION("fit", "[fit]") {
        auto start{std::chrono::steady_clock::now()};
        Results* x{model.fit(method)};
        auto end{std::chrono::steady_clock::now()};
        utils::save_performance(filename, "fit", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }

    SECTION("predict", "[predict]") {
        Results* x{model.fit(method)};
        auto start{std::chrono::steady_clock::now()};
        utils::DataFrame predictions{model.predict()};
        auto end{std::chrono::steady_clock::now()};
        utils::save_performance(filename, "predict", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }

    SECTION("predict_is", "[predict_is]") {
        Results* x{model.fit(method)};
        auto start = std::chrono::steady_clock::now();
        utils::DataFrame predictions{model.predict_is(5, true, method)};
        auto end = std::chrono::steady_clock::now();
        utils::save_performance(filename, "predict_is", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }

    SECTION("sample", "[sample]") {
        Results* x{model.fit(method)};
        auto start{std::chrono::steady_clock::now()};
        Eigen::MatrixXd sample{model.sample()};
        auto end{std::chrono::steady_clock::now()};
        utils::save_performance(filename, "sample", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }

    SECTION("ppc", "[ppc]") {
        Results* x{model.fit(method)};
        auto start{std::chrono::steady_clock::now()};
        [[maybe_unused]] double ppc{model.ppc()};
        auto end{std::chrono::steady_clock::now()};
        utils::save_performance(filename, "ppc", method,
                                std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
        delete x;
    }
}