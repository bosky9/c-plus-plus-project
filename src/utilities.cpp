#include "utilities.hpp"

#include <sstream>

utils::DataFrame utils::parse_csv(const std::string& filename) {
    std::ifstream file(filename, std::ifstream::in);
    assert(!file.fail() && "File not found!");
    utils::DataFrame df;
    if (file.is_open()) {
        // First line (if there aren't column with name "time", simulate it incrementing by 1 the time value)
        bool found_index_col{false};
        size_t index_col{0};
        std::vector<size_t> data_cols;
        std::string line;
        if (std::getline(file, line)) {
            std::istringstream lineStream(line);
            size_t i{0};
            for (std::string cell; std::getline(lineStream, cell, ',');) {
                // Find the times column
                if (cell == "time" || cell == "\"time\"") {
                    index_col       = i;
                    found_index_col = true;
                } else {
                    // The columns without name are ignored
                    if (!cell.empty() && cell != "\"\"") {
                        data_cols.push_back(i);
                        size_t first{cell.find_first_of("abcdefghijklmnopqrstuvwxyz")};
                        cell = cell.substr(first, cell.find_last_of("abcdefghijklmnopqrstuvwxyz") - first + 1);
                        df.data_name.push_back(cell);
                    }
                }
                ++i;
            }
        }
        df.data.resize(data_cols.size());
        // Other lines
        while (std::getline(file, line)) {
            std::istringstream lineStream(line);
            bool added_index_val{false};
            size_t i{0};
            for (std::string cell; std::getline(lineStream, cell, ',');) {
                auto data_col = std::find(data_cols.begin(), data_cols.end(), i);
                if (found_index_col && i == index_col)
                    df.index.push_back(std::stod(cell));
                else if (data_col != data_cols.end()) {
                    size_t data_idx = data_col - data_cols.begin();
                    df.data.at(data_idx).push_back(std::stod(cell));
                    // If there isn't the time column, simulate it
                    if (!found_index_col && !added_index_val) {
                        df.index.push_back(static_cast<double>(df.data.at(data_idx).size()));
                        added_index_val = true;
                    }
                }
                ++i;
            }
            // Fill the empty columns of the csv file with 0
            for (; i < data_cols.size(); ++i)
                df.data.at(i).push_back(0);
        }
    }
    file.close();
    return df;
}

double utils::mean(Eigen::VectorXd v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double utils::median(Eigen::VectorXd v) {
    Eigen::Index n{v.size() / 2};
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

double utils::percentile(Eigen::VectorXd v, uint8_t p) {
    std::sort(v.begin(), v.end());
    Eigen::Index index{static_cast<Eigen::Index>(std::ceil(p * 0.01 * static_cast<double>(v.size())))};
    return v(index - 1);
}

double utils::max(Eigen::VectorXd v) {
    return *std::max_element(v.begin(), v.end());
}

double utils::min(Eigen::VectorXd v) {
    return *std::min_element(v.begin(), v.end());
}

std::vector<double> utils::diff(const std::vector<double>& v) {
    std::vector<double> new_v(v.size() - 1);
    for (size_t i{0}; i < new_v.size(); ++i)
        new_v.at(i) = v.at(i + 1) - v.at(i);
    return new_v;
}