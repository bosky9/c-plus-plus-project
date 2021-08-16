//
// Created by ettorec on 14/08/21.
//

#include "../../include/output/tableprinter.hpp"
TablePrinter::TablePrinter(const std::list<std::tuple<std::string, std::string, int>>& fmt, std::string sep, std::string ul) {
    for(auto const& f : fmt) {
        std::stringstream ss;
        ss << "{" << std::get<1>(f) /* key */<< ":" << std::get<2>(f) /* width */<< "}" << sep;
        _fmt.append(ss.str());

        // try_emplace should append a new key:value pair. If key already exists, no insert
        _head.try_emplace(std::get<1>(f), std::get<0>(f));

        if (ul!="") {
            std::stringstream ulss;
            for (int i = 0; i < std::get<2>(f); i++)
                ulss << ul;
            _ul.try_emplace(std::get<1>(f), ulss.str());
        }
        else
            // map to the empty string
            _ul.clear();

        _width.try_emplace(std::get<1>(f), std::get<2>(f));
    }

}

template<typename T, std::enable_if_t<is_map_str_int<T>::value, int>>
std::string TablePrinter::row(const T& data){
    std::string str_to_return;
    std::stringstream ss;

    // I want to append every value of data map
    // I get the value from the _width key
    // I append it to the returned string, with width given by _width value
    for(auto const& w : _width)
        ss << std::setw(w.second) << data.at(w.first) << std::setfill(' ') << " ";
        str_to_return.append(ss.str());

    return std::move(str_to_return);
};

template<typename T, std::enable_if_t<is_map_str_str<T>::value, int>>
std::string TablePrinter::row(const T& data){
    std::string str_to_return;
    std::stringstream ss;

    // I want to append every value of data map
    // I get the value from the _width key
    // I append it to the returned string, with width given by _width value
    for(auto const& w : _width)
        ss << data.at(w.first) << std::setfill(' ') << " ";
    str_to_return.append(ss.str());

    return std::move(str_to_return);
};

std::list<std::string> TablePrinter::_call_(const std::list<std::map<std::string /*key*/, int>>& dataList) {
    std::list<std::string> res;
    for (auto const& dl : dataList)
        res.push_back(row(dl));
    if (!_ul.empty())
        res.push_front(row(_ul));
    res.push_front(row(_head));
    return std::move(res);
};