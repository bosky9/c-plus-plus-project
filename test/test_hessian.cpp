#include "hessian.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Compute Hessian matrix", "[hessian]") {
    Eigen::VectorXd x(7);
    x << 21, 32, 45, 52, 31, 20, 17;
    std::function<double(Eigen::VectorXd)> obj_type{
            [](Eigen::VectorXd v) { return v[0] * v[0] * v[0] * 3 + 5 - v[1] * v[0] * 3; }};
    std::cout << hessian(obj_type, x) << "\n";
}