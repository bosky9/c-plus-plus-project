#include "posterior.hpp"

posterior::posterior_opt posterior::change_function_params(posterior_wo_mb function) {
    return [function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x); };
}

posterior::posterior_opt posterior::change_function_params(posterior_with_mb function) {
    return [function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x, n.value()); };
}

posterior::posterior_wo_mb posterior::reverse_function_params(posterior_opt function) {
    return [function](Eigen::VectorXd x) { return function(x, 0); };
}