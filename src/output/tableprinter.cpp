#include "output/tableprinter.hpp"

TablePrinter::TablePrinter(const std::vector<std::tuple<std::string, std::string, int>>& fmt, const std::string& sep,
                           const std::string& ul) {
    for (auto const& f : fmt) {
        std::stringstream ss;
        ss << "{" << std::get<1>(f) /* key */ << ":" << std::get<2>(f) /* width */ << "}" << sep;
        _fmt.append(ss.str());

        // try_emplace should append a new key:value pair. If key already exists, no insert
        //std::stringstream sshe;
        //sshe << std::get<0>(f);
        //for (int i = 1; i < std::get<2>(f); ++i)
         //   sshe << " ";
         _head.try_emplace(std::get<1>(f), std::get<0>(f));


        if (!ul.empty()) {
            std::stringstream ulss;
            for (int i{0}; i < std::get<2>(f); ++i)
                ulss << ul;
            _ul.try_emplace(std::get<1>(f), ulss.str());
        } else
            // map to the empty string
            _ul.clear();

        _width.try_emplace(std::get<1>(f), std::get<2>(f));
    }
}

template<typename T, std::enable_if_t<is_map_str_int<T>::value, int>>
std::string TablePrinter::row(const T& data) {
    std::string str_to_return;
    std::stringstream ss;

    // I want to append every value of data map
    // I get the value from the _width key
    // I append it to the returned string, with width given by _width value
    // setfill + setw create (w.second) empty spaces
    for (auto const& w : _width)
        ss << std::left << std::setw(w.second) << data.at(w.first) << " ";
    str_to_return.append(ss.str());

    return str_to_return;
}

template<typename T, std::enable_if_t<is_map_str_str<T>::value, int>>
std::string TablePrinter::row(const T& data) {
    std::stringstream ss;

    for(auto const& kw : _width){
        std::string temp_str{};

        if(data.find(kw.first) != data.end())
             temp_str = (static_cast<int>((data.at(kw.first).size())) > kw.second ) ?
                data.at(kw.first).substr(0, kw.second) : data.at(kw.first);

        ss  << std::left << std::setw(kw.second) << temp_str << " ";
        auto x = ss.str();
    }
    return ss.str();
}

std::string TablePrinter::operator()(const std::list<std::map<std::string /*key*/, std::string>>& dataList) {
    std::list<std::string> res;
    for (auto const& dl : dataList)
        res.push_back(row(dl));
    if (!_ul.empty())
        res.push_front(row(_ul));
    res.push_front(row(_head));
    std::string return_string;
    for (auto const& r : res)
        return_string.append(r).append("\n");
    return return_string;
}