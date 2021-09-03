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
    int sims                                             = 3;
    BBVI bbvi                                            = BBVI(neg_posterior, q, sims);

    Eigen::MatrixXd z          = Eigen::MatrixXd::Identity(2, sims);
    Eigen::MatrixXd z_t        = z.transpose();
    Eigen::VectorXd log_q      = bbvi.normal_log_q(z_t, true);
    Eigen::VectorXd log_p      = bbvi.log_p(z_t);
    Eigen::MatrixXd grad_log_q = bbvi.grad_log_q(z);
    log_q                      = log_q.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
    Eigen::MatrixXd gradient(grad_log_q.rows(), sims);
    for (Eigen::Index i = 0; i < gradient.rows(); i++)
        gradient.row(i) = grad_log_q.row(i).array() * (log_p - log_q).transpose().array();
    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(bbvi.get_approx_param_no().sum()));
    alpha_recursion(alpha0, grad_log_q, gradient, static_cast<size_t>(bbvi.get_approx_param_no().sum()));
    double var = pow((grad_log_q.array() - grad_log_q.mean()).abs(), 2).mean();
    Eigen::MatrixXd sub(gradient.cols(), gradient.rows());
    for (Eigen::Index i = 0; i < sub.rows(); i++)
        sub.row(i) = (alpha0.transpose().array() / var) * grad_log_q.transpose().row(i).array();
    Eigen::MatrixXd vectorized = gradient - sub.transpose();

    REQUIRE(bbvi.cv_gradient(z, true) == vectorized.rowwise().mean());

    // TODO: Chiamare prima run!
    //   REQUIRE(bbvi.cv_gradient(z, false) == (gradient - sub.transpose()).rowwise().mean());
}

TEST_CASE("Draw normal", "[draw_normal]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);
    bbvi.draw_normal(true);

    // FIXME:: Chiamare prima run!
    // REQUIRE(bbvi.draw_normal() == normal);
}

TEST_CASE("Draw variables", "[draw_variables]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);
    bbvi.draw_variables();
}
