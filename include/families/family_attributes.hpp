#include <string>
#include <functional>

struct FamilyAttributes {
    std::string name;
    std::function<double (double)> link;
    bool scale;
    bool shape;
    bool skewness;
    std::function<double (double)> mean_transform;
    bool cythonized;
};