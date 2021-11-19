#include "posterior.hpp"

posterior::posterior_opt posterior::change_function_params(const posterior_wo_mb& function) {
    return [function](Eigen::VectorXd x, [[maybe_unused]] std::optional<size_t> n) { return function(std::move(x)); };
}

posterior::posterior_opt posterior::change_function_params(const posterior_with_mb& function) {
    return [function](Eigen::VectorXd x, std::optional<size_t> n) { return function(std::move(x), n.value()); };
}

posterior::posterior_wo_mb posterior::reverse_function_params(const posterior_opt& function) {
    return [function](Eigen::VectorXd x) { return function(std::move(x), 0); };
}