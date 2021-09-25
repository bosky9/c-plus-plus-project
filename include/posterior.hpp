#pragma once

#include "headers.hpp"

#include <optional>

using posterior_mb = std::function<double(Eigen::VectorXd, std::optional<size_t>)>;
using posterior    = std::function<double(Eigen::VectorXd)>;

posterior_mb change_function_params(posterior function);