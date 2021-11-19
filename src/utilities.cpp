#include "utilities.hpp"

#include <sstream>

DataFrame csvToDataFrame(std::ifstream& file) {
    DataFrame df;
    std::string line;
    std::stringstream lineStream(line);
    std::string cell;
    if (file.is_open()) {
        // First line (if there aren't column with name "time", simulate it incrementing by 1 the time value)
        bool found_index_col{false};
        size_t index_col{0};
        std::vector<size_t> data_cols;
        if (std::getline(file, line)) {
            lineStream.str("");
            cell.clear();
            size_t i{0};
            while (std::getline(lineStream, cell, ',')) {
                // Find the times column
                if (cell == "time" || cell == "\"time\"") {
                    index_col       = i;
                    found_index_col = true;
                } else {
                    // The columns without name are ignored
                    if (!cell.empty() && cell != "\"\"") {
                        data_cols.push_back(i);
                        df.data_name.push_back(cell);
                    }
                }
                ++i;
            }
        }
        df.data.resize(data_cols.size());
        // Other lines
        while (std::getline(file, line)) {
            lineStream.str("");
            cell.clear();
            bool added_index_val = false;
            size_t i             = 0;
            while (std::getline(lineStream, cell, ',')) {
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
    return df;
}

DataFrame parse_csv(const std::string& file) {
    std::ifstream myfile(file);
    DataFrame df = csvToDataFrame(myfile);
    myfile.close();
    return df;
}

DataFrame parse_csv(std::ifstream& file) {
    return csvToDataFrame(file);
}
