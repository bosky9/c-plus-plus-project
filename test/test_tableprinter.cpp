#include "output/tableprinter.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

TEST_CASE("Init and use a TablePrinter", "[TablePrinter]") {
    std::list<std::tuple<std::string, std::string, int>> fmt;
    fmt.emplace_back("1", "key1", 5);
    fmt.emplace_back("1", "key2", 4);
    std::string sep = " ";
    std::string ul = "-";

    std::list<std::map<std::string, double>> map_list;
    //Please notice how every map must have all the keys inside TablePrinter
    std::map<std::string, double> m1 { {"key2", 6.06}, {"key1", 7.778},};
    std::map<std::string, double> m2 { {"key1", 21.12}, {"key2", 3.04},};
    map_list.push_back(m1);
    map_list.push_back(m2);

    TablePrinter tp{fmt, sep, ul};
    for(auto const& str : tp._call_(map_list))
        std::cout << str << "\n";

}