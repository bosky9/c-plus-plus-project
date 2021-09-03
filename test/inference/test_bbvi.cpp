#include <catch2/catch_test_macros.hpp>

#include "inference/bbvi.hpp"

TEST_CASE("Change parameters to BBVI object", "[change_parameters, current_parameters]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q                                = std::vector<Normal>();
    q.push_back(Normal());
    q.push_back(Normal(2.0, 0.5, "exp"));
    BBVI bbvi = BBVI(neg_posterior, q, 3, "ADAM", 100, 0.01, true, true);

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

// FIXME: Chiamare prima run!
/*
TEST_CASE("Crate logq components for Normal", "[create_normal_logq") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    Eigen::VectorXd z      = Eigen::VectorXd::Ones(2);
    Eigen::VectorXd means  = Eigen::Vector2d{0.0, 2.0};
    Eigen::VectorXd scales = Eigen::Vector2d{1.0, 2.5};
    REQUIRE(bbvi.create_normal_logq(z) == Mvn::logpdf(z, means, scales).sum());
}
*/

TEST_CASE("Compute cv_gradient", "[cv_gradient, normal_log_q, log_p, grad_log_q]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q                                = std::vector<Normal>{Normal(), Normal()};
    BBVI bbvi                                            = BBVI(neg_posterior, q, 3);

    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, 3);
    bbvi.cv_gradient(z, true);

    /*
    Eigen::MatrixXd z          = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd log_q      = Eigen::Vector2d{-2.33787707, -2.33787707};
    Eigen::VectorXd log_p      = Eigen::Vector2d{-0.0, -0.0};
    Eigen::MatrixXd grad_log_q = Eigen::MatrixXd(4, 2);
    grad_log_q.row(0)          = q[0].vi_score(static_cast<Eigen::VectorXd>(z.row(0)), 0);
    grad_log_q.row(1)          = q[0].vi_score(static_cast<Eigen::VectorXd>(z.row(0)), 1);
    grad_log_q.row(2)          = q[1].vi_score(static_cast<Eigen::VectorXd>(z.row(1)), 0);
    grad_log_q.row(3)          = q[1].vi_score(static_cast<Eigen::VectorXd>(z.row(1)), 1);
    Eigen::VectorXd gradient   = grad_log_q * (log_p - log_q);
    Eigen::VectorXd alpha0     = Eigen::VectorXd::Zero(4);
    // gradient è un VETTORE e non una matrice!
    alpha_recursion(alpha0, grad_log_q, gradient, 4);
    double var                 = pow((grad_log_q.array() - grad_log_q.mean()).abs(), 2).mean();
    Eigen::VectorXd vectorized = gradient - ((alpha0 / var) * grad_log_q.transpose()).transpose();

    REQUIRE(bbvi.cv_gradient(z, true) == vectorized.colwise().mean());*/
    // TODO: Chiamare prima run!
    //   REQUIRE(bbvi.cv_gradient(z, false) == vectorized.colwise().mean());
}

TEST_CASE("Draw normal", "[draw_normal]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    auto means_scales = bbvi.get_means_and_scales_from_q();
    Eigen::MatrixXd normal(3, means_scales.first.size());
    normal.row(0) = Mvn::random(means_scales.first[0], means_scales.second[0], means_scales.first.size());
    normal.row(1) = Mvn::random(means_scales.first[1], means_scales.second[1], means_scales.first.size());
    normal        = normal.transpose();
    REQUIRE(bbvi.draw_normal(true) == normal);

    // FIXME:: Chiamare prima run!
    /*
    means_scales = bbvi.get_means_and_scales();
    normal.row(0) = Mvn::random(means_scales.first[0], means_scales.second[0], means_scales.first.size());
    normal.row(1) = Mvn::random(means_scales.first[1], means_scales.second[1], means_scales.first.size());
    normal        = normal.transpose();
    REQUIRE(bbvi.draw_normal() == normal);
     */
}

TEST_CASE("Draw variables", "[draw_variables]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    Eigen::MatrixXd z(q.size(), 3);
    z.row(0) = q[0].draw_variable_local(3);
    z.row(1) = q[1].draw_variable_local(3);

    REQUIRE(bbvi.draw_variables() == z);
}