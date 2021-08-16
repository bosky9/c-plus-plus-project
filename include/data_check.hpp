#pragma once

#include <iterator>
#include <memory>
#include <numeric>

#include "headers.hpp"

/**
 * @brief Struct containing the data returned by data_check
 * @see data_check
 */
struct CheckedData {
    std::unique_ptr<std::vector<double>> transformed_data; ///< Raw data array for use in the model
    std::unique_ptr<std::string> data_name;                ///< Name of the data
    std::unique_ptr<std::vector<size_t>> data_index;       ///< The time indices for the data
};

/**
 * @brief Struct containing the data returned by mv_data_check
 * @see data_check
 */
struct CheckedDataMv {
    std::unique_ptr<std::vector<std::vector<double>>> transformed_data; ///< Raw data array for use in the model
    std::unique_ptr<std::vector<size_t>> data_name;                     ///< Names of the data
    std::unique_ptr<std::vector<size_t>> data_index;                    ///< The time indices for the data
};

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 */
template<typename T>
std::unique_ptr<CheckedData> data_check(std::vector<T>& data) {
    static_assert(std::is_floating_point_v<T>,
                  "data_check accepts as data only a vector of floating points or a vector containing vectors of "
                  "floating points");

    std::unique_ptr<CheckedData> checked_data(new CheckedData());
    checked_data->transformed_data.reset(new std::vector<double>{data});
    checked_data->data_index.reset(new std::vector<size_t>(data.size()));
    std::iota(checked_data->data_index->begin(), checked_data->data_index->end(), 0);
    checked_data->data_name = std::make_unique<std::string>("Series");

    return checked_data;
}

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @param target Target column
 * @return A struct containing the transformed data, relative name and indices
 */
template<typename T>
std::unique_ptr<CheckedData> data_check(std::vector<std::vector<T>>& data, size_t target) {
    static_assert(std::is_floating_point_v<T>,
                  "data_check accepts as data only a vector of floating points or a vector containing vectors of "
                  "floating points");

    std::unique_ptr<CheckedData> checked_data(new CheckedData());

    if (target == NULL) {
        checked_data->transformed_data.reset(new std::vector<double>{data[0]});
        checked_data->data_index.reset(new std::vector<size_t>(data[0].size()));
    } else {
        checked_data->transformed_data.reset(new std::vector<double>{data[target]});
        checked_data->data_index.reset(new std::vector<size_t>(data[target].size()));
    }

    std::iota(checked_data->data_index->begin(), checked_data->data_index->end(), 0);
    checked_data->data_name = std::make_unique<std::string>("Series");

    return checked_data;
}

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 */
template<typename T>
std::unique_ptr<CheckedDataMv> mv_data_check(std::vector<std::vector<T>>& data) {
    static_assert(std::is_floating_point_v<T>,
                  "data_check accepts as data only a vector of floating points or a vector containing vectors of "
                  "floating points");

    std::unique_ptr<CheckedDataMv> checked_data(new CheckedDataMv());
    checked_data->transformed_data =
            std::make_unique<std::vector<std::vector<double>>>(std::vector<std::vector<double>>{data});
    checked_data->data_index = std::make_unique<std::vector<size_t>>(data[0].size());
    std::iota(checked_data->data_index->begin(), checked_data->data_index->end(), 0);
    checked_data->data_name = std::make_unique<std::vector<size_t>>(data[0].size());
    std::iota(checked_data->data_name->begin(), checked_data->data_name->end(), 1);

    return checked_data;
}