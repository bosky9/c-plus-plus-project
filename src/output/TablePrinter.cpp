//
// Created by ettorec on 14/08/21.
//

#include "../../include/output/tableprinter.hpp"
TablePrinter::TablePrinter(const std::list<std::tuple<std::string, std::string, int>>& fmt, std::string sep, std::string ul) {
    for(auto const& f : fmt) {
        std::stringstream ss;
        ss << "{" << std::get<1>(f) << "}:{" << std::get<2>(f) << "}" << sep;
        _fmt.append(ss.str());

        _head = std::map<std::string, std::string>{ {std::get<1>(f), std::get<0>(f)} };

        if (ul!="") {
            std::stringstream ulss;
            for (int i = 0; i<std::get<2>(f); i++)
                ulss << ul;
            _ul = std::map<std::string, std::string>{ {std::get<1>(f), ulss.str()} };
        }
        else
            // map to the empty string
            _ul = std::map<std::string, std::string>{ {std::get<1>(f), ul} };

        _width = std::map<std::string, int>{ {std::get<1>(f), std::get<2>(f)} };
    }

}