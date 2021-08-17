#pragma once
#include "../headers.hpp"
#include <tuple>
#include <map>
#include <sstream>
#include <cstdio>
#include <iomanip>

//@Todo: find a better way to implement key

template< typename T >
struct is_map_str_int{
    static const bool value = false;
};

template<>
struct is_map_str_int<std::map<std::basic_string<char>, double>>{
    static const bool value = true;
};

template< typename T >
struct is_map_str_str{
    static const bool value = false;
};

template<>
struct is_map_str_str<std::map<std::string, std::string>>{
    static const bool value = true;
};

class TablePrinter {
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
    TablePrinter(const std::list<std::tuple<std::string, std::string, int>>& fmt,
                 const std::string& sep=" ",
                 const std::string& ul="");

    /**
     * \brief Appends every value of data map in a string; the value is taken using the @var _width key.
     * \brief Width of every string is given by @var _width value.
     * @tparam T: could be a map of str:double or a map of str:str
     * @param data: a map of keys (str) and values (double)
     * @return string, where all the values are appended together
     */
    template<typename T, std::enable_if_t<is_map_str_int<T>::value, int> = 0>
    std::string row(const T& data); //SFINAE

    template<typename T, std::enable_if_t<is_map_str_str<T>::value, int> = 0>
    std::string row(const T& data); //SFINAE

    std::list<std::string> _call_(const std::list<std::map<std::string /*key*/, double>>& dataList);

};