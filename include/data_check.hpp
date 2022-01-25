#pragma once

#include "utilities.hpp"

#include <vector>

/**
 * @brief Checks data type
 * @param data Field to specify the time series data that will be used
 * @return A struct containing the transformed data, relative name and indices
 *
 * @details Represents the Python case where a np.array or a pd.DataFrame is passed to the function.
 */
template<typename T>
utils::SingleDataFrame data_check(const T& data, std::vector<double>& data_original,
                                  const std::optional<std::string>& target = std::nullopt);

template<>
utils::SingleDataFrame data_check<std::vector<double>>(const std::vector<double>& data,
                                                       std::vector<double>& data_original,
                                                       const std::optional<std::string>& target);

template<>
utils::SingleDataFrame data_check<utils::DataFrame>(const utils::DataFrame& data, std::vector<double>& data_original,
                                                    const std::optional<std::string>& target);
