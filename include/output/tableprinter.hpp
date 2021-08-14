#pragma once
#include "../headers.hpp"
#include <tuple>
#include <map>

class TablePrinter() {
private:
    std::map head;
    std::map ul;
    std::map width;
public:
    /**
    @param fmt: list of tuple(heading, key, width)
    heading: str, column label
    key: dictionary key to value to print
    width: int, column width in chars
    @param sep: string, separation between columns
    @param ul: string, character to underline column label, or None for no underlining
    */
    TablePrinter(std::tuple<std::string, std::string, int> fmt, std::string sep=" ", std::string ul="");

    std::string* row();

    std::string** _call_();

}