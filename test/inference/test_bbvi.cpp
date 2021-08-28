#include <catch2/catch_test_macros.hpp>

#include "inference/bbvi.hpp"

TEST_CASE("Change parameters to BBVI object", "[change_parameters, current_parameters]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q                                = std::vector<Normal>();
    q.push_back(Normal());
    q.push_back(Normal(2.0, 0.5, "exp"));
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    Eigen::VectorXd params = static_cast<Eigen::VectorXd>(Eigen::Vector4d{3.0, 1.0, 2.0, 1.0});
    bbvi.change_parameters(params);

    q[0].vi_change_param(0, 3.0);
    q[0].vi_change_param(1, 1.0);
    q[1].vi_change_param(0, 3.0);
    q[1].vi_change_param(1, 1.0);
    // REQUIRE(bbvi.get_q()[0] == q[0]);
    // REQUIRE(bbvi.get_q()[1] == q[1]);
    REQUIRE(bbvi.current_parameters() == params);
}

/*
TEST_CASE("Compute cv_gradient", "[cv_gradient]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q                                = std::vector<Normal>{Normal()};
    BBVI bbvi                                            = BBVI(neg_posterior, q, 3);

    Eigen::VectorXd z                                    = Eigen::VectorXd::Ones(3);
    bbvi.cv_gradient(z);
}*/