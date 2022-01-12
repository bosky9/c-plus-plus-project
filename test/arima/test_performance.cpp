/**
 * @file test_performance.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "arima/arima.hpp"

#include "utilities.hpp" // utils::parse_csv

#include <catch2/catch_test_macros.hpp>
#include <chrono>   // std::chrono::steady_clock::now, std::chrono::duration_cast
#include <iostream> // std::cout

TEST_CASE("Test performances on sunspot data: fit", "[fit]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    auto start = std::chrono::steady_clock::now();
    ARIMA model{data, 2, 2, 0, "sunactivity"};
    Results* x{model.fit("BBVI")};
    auto end = std::chrono::steady_clock::now();

    std::cout << "\nElapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " sec\n";
    std::cout << "Elapsed time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    delete x;
}

TEST_CASE("Test performances on sunspot data: predict", "[predict]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    auto start = std::chrono::steady_clock::now();
    ARIMA model{data, 2, 2, 0, "sunactivity"};
    Results* x{model.fit("BBVI")};
    utils::DataFrame predictions{model.predict()};
    auto end = std::chrono::steady_clock::now();

    std::cout << "\nElapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " sec\n";
    std::cout << "Elapsed time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    delete x;
}

TEST_CASE("Test performances on sunspot data: predict_is", "[predict_is]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    auto start = std::chrono::steady_clock::now();
    ARIMA model{data, 2, 2, 0, "sunactivity"};
    Results* x{model.fit("BBVI")};
    utils::DataFrame predictions{model.predict_is()};
    auto end = std::chrono::steady_clock::now();

    std::cout << "\nElapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " sec\n";
    std::cout << "Elapsed time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    delete x;
}

TEST_CASE("Test performances on sunspot data: sample", "[sample]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    auto start = std::chrono::steady_clock::now();
    ARIMA model{data, 2, 2, 0, "sunactivity"};
    Results* x{model.fit("BBVI")};
    Eigen::MatrixXd sample{model.sample()};
    auto end = std::chrono::steady_clock::now();

    std::cout << "\nElapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " sec\n";
    std::cout << "Elapsed time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    delete x;
}

TEST_CASE("Test performances on sunspot data: ppc", "[ppc]") {
    utils::DataFrame data{utils::parse_csv("../data/sunspots.csv")};

    auto start = std::chrono::steady_clock::now();
    ARIMA model{data, 2, 2, 0, "sunactivity"};
    Results* x{model.fit("BBVI")};
    [[maybe_unused]] double ppc{model.ppc()};
    auto end = std::chrono::steady_clock::now();

    std::cout << "\nElapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " sec\n";
    std::cout << "Elapsed time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    delete x;
}