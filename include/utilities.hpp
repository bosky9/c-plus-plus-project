#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

/**
 * @brief Struct that represents the input data for a time-series model
 */
struct DataFrame final {
    std::vector<double> index;             ///< The times of the input data (years, days or seconds)
    std::vector<std::vector<double>> data; ///< The univariate time series data (values) that will be used
    std::vector<std::string> data_name;    ///< The names of the data
};

DataFrame parse_csv(const std::string& path);