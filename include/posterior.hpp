/**
 * @file posterior.hpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#pragma once

#include "Eigen/Core" // Eigen::VectorXd

#include <functional> // std::function
#include <optional>   // std::optional

namespace posterior {

using posterior_opt     = std::function<double(Eigen::VectorXd, std::optional<size_t>)>;
using posterior_wo_mb   = std::function<double(Eigen::VectorXd)>;
using posterior_with_mb = std::function<double(Eigen::VectorXd, size_t)>;

/**
 * @param function A function, which will take only a vector.
 * @return  The function, which will now take an optional parameter but will not consider it.
 */
posterior_opt change_function_params(const posterior_wo_mb& function);

/**
 * @param function A function which will take a vector and an integer parameter.
 * @return  The function, taking in account the optional parameter.
 */
posterior_opt change_function_params(const posterior_with_mb& function);

/**
 * @param function A function with an optional integer parameter.
 * @return  The function, without the optional parameter.
 */
posterior_wo_mb reverse_function_params(const posterior_opt& function);

} // namespace posterior