/**
 * @file tableprinter.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include <list>   // std::list
#include <map>    // std::map
#include <tuple>  // std::tuple
#include <vector> // std::vector

template<typename T>
struct is_map_str_int final {
    static const bool value = false;
};

template<>
struct is_map_str_int<std::map<std::basic_string<char>, double>> final {
    static const bool value = true;
};

template<typename T>
struct is_map_str_str final {
    static const bool value = false;
};

template<>
struct is_map_str_str<std::map<std::string, std::string>> final {
    static const bool value = true;
};

class TablePrinter final {
private:
    std::string _fmt;
    std::map<std::string, std::string> _head;
    std::map<std::string, std::string> _ul;
    std::map<std::string, int> _width;

public:
    /**
    @param fmt: list of tuple(heading, key, width)
    heading: str, column label
    key: str (arbitrary), dictionary key to value to print
    width: int, column width in chars
    @param sep: string, separation between columns
    @param ul: string, character to underline column label, or None for no underlining
    */
    explicit TablePrinter(const std::vector<std::tuple<std::string, std::string, int>>& fmt,
                          const std::string& sep = " ", const std::string& ul = "");

    /**
     * @brief Appends every value of data map in a string; the value is taken using the @var _width key.
     * @brief Width of every string is given by @var _width value.
     * @tparam T: could be a map of str:double or a map of str:str
     * @param data: a map of keys (str) and values (double or string)
     * @return string, where all the values are appended together
     */
    template<typename T, std::enable_if_t<is_map_str_int<T>::value, int> = 0>
    std::string row(const T& data); // SFINAE

    template<typename T, std::enable_if_t<is_map_str_str<T>::value, int> = 0>
    std::string row(const T& data); // SFINAE

    std::string operator()(const std::list<std::map<std::string /*key*/, std::string>>& dataList);
};