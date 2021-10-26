#include "posterior.hpp"

posterior change_function_params(posterior_wo_mb function) {
    return [function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x); };
}

posterior change_function_params(posterior_with_mb function) {
    return [function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x, n.value()); };
}

posterior_wo_mb reverse_function_params(posterior function) {
    return [function](Eigen::VectorXd x) { return function(x, 0); };
}