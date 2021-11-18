#pragma once

#include "headers.hpp"

#include <optional>

namespace posterior {

using posterior_opt     = std::function<double(Eigen::VectorXd, std::optional<size_t>)>;
using posterior_wo_mb   = std::function<double(Eigen::VectorXd)>;
using posterior_with_mb = std::function<double(Eigen::VectorXd, size_t)>;

/**
 * @param function A function, which will take only a vector.
 * @return  The function, which will now take an optional parameter but will not consider it.
 */
posterior_opt change_function_params(posterior_wo_mb function);

/**
 * @param function A function which will take a vector and an integer parameter.
 * @return  The function, taking in account the optional parameter.
 */
posterior_opt change_function_params(posterior_with_mb function);

/**
 * @param function A function with an optional integer parameter.
 * @return  The function, without the optional parameter.
 */
posterior_wo_mb reverse_function_params(posterior_opt function);

} // namespace posterior