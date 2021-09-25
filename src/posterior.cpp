#include "posterior.hpp"

posterior_mb change_function_params(posterior function) {
    return posterior_mb{[function](Eigen::VectorXd x, std::optional<size_t> n) { return function(x); }};
}