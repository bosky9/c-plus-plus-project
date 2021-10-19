#pragma once

#include "headers.hpp"

#include <optional>

using posterior         = std::function<double(Eigen::VectorXd, std::optional<size_t>)>;
using posterior_wo_mb   = std::function<double(Eigen::VectorXd)>;
using posterior_with_mb = std::function<double(Eigen::VectorXd, size_t)>;

posterior change_function_params(posterior_wo_mb function);

posterior change_function_params(posterior_with_mb function);