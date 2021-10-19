#include "posterior.hpp"

posterior change_function_params(posterior_wo_mb function) {
    return posterior{[function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x); }};
}

posterior change_function_params(posterior_with_mb function) {
    return posterior{[function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x, n.value()); }};
}