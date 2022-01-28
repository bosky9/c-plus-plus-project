/**
 * @file covariances.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd

#include <algorithm>   // std::transform
#include <cassert>     // static_assert
#include <iterator>    // std::back_inserter
#include <numeric>     // std::inner_product, std::accumulate
#include <type_traits> // std::is_floating_point
#include <vector>      // std::vector

namespace covariances {
/**
 * @brief Calculate the sample autocovariance of two arrays (stationarity assumed)
 * @param x Array of data
 * @param lag Index to split the array into two arrays with equal sizes
 * @return The sample autocovariance of the two arrays
 *
 * @details
        A back-insert iterator is a special type of output iterator
        designed to allow algorithms that usually overwrite elements (such as copy)
        to instead insert new elements automatically at the end of the container.

        Used to expand vector x1, x2.
 */
template<typename T, int N>
T cov(const Eigen::Vector<T, N>& x, size_t lag = 0) {
    static_assert(std::is_floating_point<T>::value, "cov accepts only vector of float or double");
    assert(lag < static_cast<size_t>(x.size()));
    const T mean = std::accumulate(x.begin(), x.end(), (T) 0) / x.size();
    std::vector<T> x1, x2;
    std::transform(x.begin() + lag, x.end(), std::back_inserter(x1), [mean](T val) { return val - mean; });
    std::transform(x.begin(), x.end() - lag, std::back_inserter(x2), [mean](T val) { return val - mean; });
    assert(x1.size() == x2.size());
    T inner = std::inner_product(x1.begin(), x1.end(), x2.begin(), (T) 0);
    return inner / (T) x1.size();
}

/**
 * @brief Calculate the sample autocorrelation function of an array (stationarity assumed)
 * @param x Array of data
 * @param lag Index to split the array into two arrays with equal sizes
 * @return The sample autocorrelation function of the array x
 */
template<typename T, int N>
double acf(const Eigen::Vector<T, N>& x, size_t lag = 0) {
    static_assert(std::is_floating_point<T>::value, "acf accepts only vector of float or double");
    return cov(x, lag) / cov(x);
}
} // namespace covariances