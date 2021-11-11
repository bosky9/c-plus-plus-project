#pragma once

#include "families/family.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Struct that represents the input data for a time-series model
 */
struct DataFrame final {
    std::vector<double> index;             ///< The times of the input data (years, days or seconds)
    std::vector<std::vector<double>> data; ///< The univariate time series data (values) that will be used
    std::vector<std::string> data_name;    ///< The names of the data
};

DataFrame parse_csv(const std::string& path);

DataFrame parse_csv(std::ifstream& path);

/**
 * @brief Returns true if is the object is an instance of subclass T (of Family)
 * @tparam T Subclass to check
 * @param obj Pointer to an object of the superclass
 * @return If is the object is an instance of subclass T
 */
template<typename T>
inline bool instanceof (Family * obj) {
    return obj == dynamic_cast<T*>(obj);
}
