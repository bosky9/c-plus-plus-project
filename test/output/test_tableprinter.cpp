#include "output/tableprinter.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

using namespace std;

// Function which return string by concatenating it.
string repeat(string s, int n)
{
    // Copying given string to temporary string.
    string s1 = s;

    for (int i=1; i<n;i++)
        s += s1; // Concatenating strings

    return s;
}

TEST_CASE("Init and use a TablePrinter", "[TablePrinter]") {
    std::vector<std::tuple<std::string, std::string, int>> fmt;
    // Head is the header of each column
    // Each column has a key associated with it
    // The third param is the number of chars. of the values of each column
    fmt.emplace_back("ClassID", "classid", 11);
    fmt.emplace_back("Dept", "dept", 8);
    fmt.emplace_back("Course Number", "coursenum", 20);
    fmt.emplace_back("Area", "area", 8);
    fmt.emplace_back("Title", "title", 30);
    std::string sep = " ";
    std::string ul  = "=";

    std::list<std::map<std::string, std::string>> map_list;
    // Please notice how every map must have all the keys inside TablePrinter
    std::map<std::string, std::string> m1{
        {"classid", "foo"},
        {"dept", "bar"},
        {"coursenum", "foo"},
        {"area", "bar"},
        {"title", "foo"}
    };
    std::map<std::string, std::string> m2{
        {"classid", "yoo"},
        {"dept", "hat"},
        {"coursenum", "yoo"},
        {"area", "bar"},
        {"title", "hat"}
    };
    std::map<std::string, std::string> m3{
        {"classid", repeat("yoo", 9)},
        {"dept", repeat("hat", 9)},
        {"coursenum", repeat("yoo", 9)},
        {"area", repeat("bar", 9)},
        {"title", repeat("hat", 9)}
    };
    map_list.push_back(m1);
    map_list.push_back(m2);
    map_list.push_back(m3);


    TablePrinter tp{fmt, sep, ul};
    // for(auto const& str : tp._call_(map_list))
    //     std::cout << str << "\n";
    std::cout << tp(map_list) << "\n";
}