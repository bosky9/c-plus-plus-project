/**
 * @file utilities.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core"          // Eigen::VectorXd
#include "families/family.hpp" // Family
#include "families/flat.hpp"   // Flat
#include "families/normal.hpp" // Normal
#include "inference/bbvi.hpp"  // BBVI

#include <chrono> // std::chrono::milliseconds
#include <string> // std::string
#include <vector> // std::vector

namespace utils {

/**
 * @brief Struct that represents the internal data of a time-series model
 */
struct SingleDataFrame final {
    std::vector<double> index; ///< The times of the input data (years, days or seconds)
    std::vector<double> data;  ///< The univariate time series data (values) that will be used
    std::string data_name;     ///< The names of the data
};

/**
 * @brief Struct that represents the input data for a time-series model
 */
struct DataFrame final {
    std::vector<double> index;             ///< The times of the input data (years, days or seconds)
    std::vector<std::vector<double>> data; ///< The univariate time series data (values) that will be used
    std::vector<std::string> data_name;    ///< The names of the data
};

/**
 * @brief Returns a DataFrame from a .csv file
 * @param filename String rapresenting the filename of the file
 * @return DataFrame obtained from values in the file
 */
DataFrame parse_csv(const std::string& filename);

/**
 * @brief Create file to write perfomance results
 * @param test
 */
std::string create_performance_file(const std::string& data, const std::string& method, size_t ar, size_t ma,
                                    size_t integ = 0, const std::string& family = "Normal",
                                    std::optional<std::string> target = std::nullopt);

/**
 * @brief Save performance tests to csv file performance_results.csv
 * @test_name Name of the function invoked
 * @method Name of the fit method used
 * @time Elapsed time in milliseconds
 * @print If true prints also to stdout
 */
void save_performance(const std::string& filename, const std::string& test_name, const std::string& method,
                      std::chrono::milliseconds time, bool print = false);

/**
 * @brief Check if the class R of object is a subclass of T
 * @tparam T Superclass
 * @tparam R Subclass
 * @param obj Object of class R
 * @return If R is a subclass of T
 */
template<typename T, typename R>
bool isinstance(const R* obj) {
    return obj == dynamic_cast<const T*>(obj);
}

/**
 * @brief Mean function applied to a vector
 * @param v Vector of double
 * @return Mean of values inside the vector
 */
double mean(Eigen::VectorXd v);

/**
 * @brief Median function applied to a vector
 * @param v Vector of double
 * @return Median of values inside the vector
 */
double median(Eigen::VectorXd v);

/**
 * @brief Computes the percentile of a given vector
 * @param v Vector of doubles
 * @param p Percentile value
 * @return Percentile p of v
 */
double percentile(Eigen::VectorXd v, uint8_t p);

/**
 * @brief Max function applied to a vector
 * @param v Vector of double
 * @return Maximum value inside the vector
 */
double max(Eigen::VectorXd v);

/**
 * @brief Min function applied to a vector
 * @param v Vector of double
 * @return Minimum value inside the vector
 */
double min(Eigen::VectorXd v);

/**
 * @brief Compute difference between subsequent values in the vector
 * @param v Vector of double
 * @return Vector of differences
 */
std::vector<double> diff(const std::vector<double>& v);

/**
 * @brief Utility for comparing values in sample method (ARIMA)
 * @param a First value
 * @param b Second value
 * @return If the absolute value of the first one is greater than the one of the second
 */
bool abs_compare(double a, double b);
} // namespace utils
