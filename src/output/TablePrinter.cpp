//
// Created by ettorec on 14/08/21.
//

#include "../../include/output/tableprinter.hpp"
TablePrinter::TablePrinter(const std::list<std::tuple<std::string, std::string, int>>& fmt, std::string sep, std::string ul) {
    for(auto const& f : fmt) {
        std::stringstream ss;
        ss << "{" << std::get<1>(f) << "}:{" << std::get<2>(f) << "}" << sep;
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
            _ul.try_emplace(std::get<1>(f), ul);

        _width.try_emplace(std::get<1>(f), std::get<2>(f));
    }

}