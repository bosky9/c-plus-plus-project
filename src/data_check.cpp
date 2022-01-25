#include "data_check.hpp"

#include "utilities.hpp"

#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

template<>
utils::SingleDataFrame data_check<std::vector<double>>(const std::vector<double>& data,
                                                       std::vector<double>& data_original,
                                                       [[maybe_unused]] const std::optional<std::string>& target) {
    utils::SingleDataFrame checked_data;
    checked_data.data  = data;
    checked_data.index = std::vector<double>(data.size());
    std::iota(checked_data.index.begin(), checked_data.index.end(), 0);
    checked_data.data_name = "Series";
    data_original          = data;
    return checked_data;
}

template<>
utils::SingleDataFrame data_check<utils::DataFrame>(const utils::DataFrame& data, std::vector<double>& data_original,
                                                    const std::optional<std::string>& target) {
    assert(data.data_name.size() == data.data.size());

    utils::SingleDataFrame checked_data;
    if (!target.has_value() || target.value().empty()) {
        checked_data.data      = data.data.at(0);
        checked_data.data_name = data.data_name.at(0);
        data_original          = data.data.at(0);
    } else {
        auto it{std::find(data.data_name.begin(), data.data_name.end(), target)};
        assert(it != data.data_name.end());
        assert(data.data_name.size() == data.data.size());
        size_t col = it - data.data_name.begin();
        assert(data.data.at(col).size() == data.index.size());

        checked_data.data      = data.data.at(col);
        checked_data.data_name = target.value();
        data_original          = data.data.at(col);
    }
    checked_data.index = data.index;

    return checked_data;
}