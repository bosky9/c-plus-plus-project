#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "headers.hpp"

template<typename T>
T cov(const std::vector<T>& x, size_t lag = 0) {
    static_assert(std::is_floating_point<T>::value, "cov accepts only float,double types");
    assert(lag < x.size());
    const T mean = accumulate(x.begin(), x.end(), 0.0) / x.size();
    std::vector<T> x1, x2;
    std::transform(x.begin() + lag, x.end(), std::back_inserter(x1), [mean](T val) { return val - mean; });
    std::transform(x.begin(), x.end() - lag, std::back_inserter(x2), [mean](T val) { return val - mean; });
    assert(x1.size() == x2.size());
    T inner = std::inner_product(x1.begin(), x1.end(), x2.begin(), (T) 0);
    return inner / (T) x1.size();
}

template<typename T>
T acf(const std::vector<T>& x, size_t lag = 0) {
    static_assert(std::is_floating_point<T>::value, "acf accepts only float,double types");
    return cov(x, lag) / cov(x);
}

// TODO: missing function acf_plot