#pragma once

#include "utilities.hpp"

#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Represents the Python case where a np.array is passed to the function, that identifies already the final
 * data to process.
 */
inline utils::SingleDataFrame data_check(const std::vector<double>& data) {
    utils::SingleDataFrame checked_data;
    checked_data.data  = data;
    checked_data.index = std::vector<double>(data.size());
    std::iota(checked_data.index.begin(), checked_data.index.end(), 0);
    checked_data.data_name = "Series";
    return checked_data;
}

/**
 * @brief Checks data type
 * @param data_frame Input data for the time-series model (including data, index and column names)
 * @param target Target column
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Represents the Python case where a pd.DataFrame is passed to the function.
 */
inline utils::SingleDataFrame data_check(const utils::DataFrame& data_frame, const std::string& target = "") {
    assert(data_frame.data_name.size() == data_frame.data.size());

    utils::SingleDataFrame checked_data;
    if (target.empty()) {
        checked_data.data      = data_frame.data.at(0);
        checked_data.data_name = data_frame.data_name.at(0);
    } else {
        auto it{std::find(data_frame.data_name.begin(), data_frame.data_name.end(), target)};
        assert(it != data_frame.data_name.end());
        assert(data_frame.data_name.size() == data_frame.data.size());
        size_t col = it - data_frame.data_name.begin();
        assert(data_frame.data.at(col).size() == data_frame.index.size());

        checked_data.data      = data_frame.data.at(col);
        checked_data.data_name = target;
    }
    checked_data.index = data_frame.index;

    return checked_data;
}

// TODO: The following methods (used only in VAR models) is useful? Consider to remove it
/**
 * @brief Checks data type
 * @param data_frame Input data for the time-series model
 * @param target Target column
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Represents the Python case where a np.array is passed to the function.
 */
inline utils::DataFrame mv_data_check(const std::vector<std::vector<double>>& data) {
    utils::DataFrame checked_data;
    checked_data.data = data;
    checked_data.index.resize(data.at(0).size());
    std::iota(checked_data.index.begin(), checked_data.index.end(), 0);
    checked_data.data_name.resize(data.at(0).size());
    std::transform(checked_data.index.begin(), checked_data.index.end(), checked_data.data_name.begin(),
                   [](double x) { return std::to_string(static_cast<int>(x) + 1); });

    return checked_data;
}

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Represents the Python case where a pd.utils::DataFrame is passed to the function.
 */
inline utils::DataFrame mv_data_check(const utils::DataFrame& data_frame) {
    return {data_frame};
}