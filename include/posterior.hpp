#pragma once

#include "headers.hpp"

#include <optional>

namespace posterior {

using posterior_opt     = std::function<double(Eigen::VectorXd, std::optional<size_t>)>;
using posterior_wo_mb   = std::function<double(Eigen::VectorXd)>;
using posterior_with_mb = std::function<double(Eigen::VectorXd, size_t)>;

posterior_opt change_function_params(posterior_wo_mb function);

posterior_opt change_function_params(posterior_with_mb function);

posterior_wo_mb reverse_function_params(posterior_opt function);

}; // namespace posterior