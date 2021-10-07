#pragma once

#include <iterator>
#include <map>
#include <memory>
#include <numeric>

#include "headers.hpp"

/**
 * @brief Struct containing the data returned by data_check
 * @see data_check
 */
struct CheckedData final {
    std::vector<double> transformed_data; ///< Raw data array for use in the model
    std::vector<std::string> data_name;   ///< Name of the data
    std::vector<double> data_index;       ///< The time indices for the data
};

/**
 * @brief Struct containing the data returned by mv_data_check
 * @see data_check
 */
struct CheckedDataMv final {
    std::vector<std::vector<double>> transformed_data; ///< Raw data array for use in the model
    std::vector<size_t> data_name;                     ///< Names of the data
    std::vector<double> data_index;                    ///< The time indices for the data
};

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Having a template function which takes an std::vector as an input
 *          is necessary to cover the python case where a np.array is passed
 *          to the function.
 */
template<typename T>
CheckedData data_check(const std::vector<T>& data, const std::vector<double>& index) {
    static_assert(std::is_floating_point_v<T>,
                  "data_check accepts as data only a vector of floating points or a vector containing vectors of "
                  "floating points");
    assert(data.size() == index.size());

    CheckedData cd;
    cd.transformed_data = data;
    cd.data_index = index;
    cd.data_name = {"Series"};
    return std::move(cd);
}

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @param target Target column
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Using a map to approximate the python case where
 *          a pd.DataFrame is passed to the function.
 */
template<typename T>
CheckedData data_check(const std::map<std::string, std::vector<T>>& data, const std::vector<double>& index,
                        const std::string& target) {
    static_assert(std::is_floating_point_v<T>,
                  "data_check accepts as data only a vector of floating points or a vector containing vectors of "
                  "floating points");
    assert(data.at(target).size() == index.size());

    CheckedData cd;
    cd.transformed_data = data.at(target);
    cd.data_index = index;
    cd.data_name = {target};
    return std::move(cd);
}

// TODO: The following method (used only in VAR models) is useful? Consider to remove it

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 */
template<typename T>
CheckedDataMv mv_data_check(const std::vector<std::vector<T>>& data) {
    static_assert(std::is_floating_point_v<T>,
                  "data_check accepts as data only a vector of floating points or a vector containing vectors of "
                  "floating points");

    CheckedDataMv cd;
    cd.transformed_data = data;
    cd.data_index.resize(data.at(0).size());
    std::iota(cd.data_index.begin(), cd.data_index.end(), 0);
    cd.data_name.resize(data[0].size());
    std::iota(cd.data_name.begin(), cd.data_name.end(), 1);

    return std::move(cd);
}