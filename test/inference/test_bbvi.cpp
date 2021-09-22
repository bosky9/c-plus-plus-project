#include <catch2/catch_test_macros.hpp>

#include "inference/bbvi.hpp"

TEST_CASE("Create a BBVI object", "[BBVI]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q                                = {Normal()};
    BBVI bbvi1                                           = BBVI(neg_posterior, q, 3, "ADAM", 100, 0.01, false, false);
    BBVI bbvi2{bbvi1};
    REQUIRE(bbvi2 == bbvi1);
    BBVI bbvi3{std::move(bbvi1)};
    REQUIRE(bbvi3 == bbvi2);
    BBVI bbvi4 = BBVI(neg_posterior, q, 0);
    bbvi4      = bbvi2;
    bbvi4      = bbvi3;
    REQUIRE(bbvi4 == bbvi2);
    bbvi4 = std::move(bbvi2);
    REQUIRE(bbvi4 == bbvi3);
}

TEST_CASE("Change parameters to BBVI object", "[change_parameters, current_parameters]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return 0; };
    std::vector<Normal> q                                = std::vector<Normal>();
    q.emplace_back();
    q.emplace_back(2.0, 0.5, "exp");
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    Eigen::VectorXd params = static_cast<Eigen::VectorXd>(Eigen::Vector4d{3.0, 1.0, 2.0, 1.0});
    bbvi.change_parameters(params);

    q[0].vi_change_param(0, 3.0);
    q[0].vi_change_param(1, 1.0);
    q[1].vi_change_param(0, 3.0);
    q[1].vi_change_param(1, 1.0);
    REQUIRE(bbvi.current_parameters() == params);
}

TEST_CASE("Compute cv_gradient", "[cv_gradient]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
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

    bbvi._optim = std::make_unique<RMSProp>(bbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                            bbvi.get_learning_rate(), 0.99);
    log_q       = bbvi.normal_log_q(z_t, false);
    log_q       = log_q.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
    for (Eigen::Index i = 0; i < gradient.rows(); i++)
        gradient.row(i) = grad_log_q.row(i).array() * (log_p - log_q).transpose().array();
    alpha_recursion(alpha0, grad_log_q, gradient, static_cast<size_t>(bbvi.get_approx_param_no().sum()));
    var = pow((grad_log_q.array() - grad_log_q.mean()).abs(), 2).mean();
    for (Eigen::Index i = 0; i < sub.rows(); i++)
        sub.row(i) = (alpha0.transpose().array() / var) * grad_log_q.transpose().row(i).array();
    vectorized = gradient - sub.transpose();
    REQUIRE(bbvi.cv_gradient(z, false) == vectorized.rowwise().mean());
}

TEST_CASE("Draw normal", "[draw_normal]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);
    bbvi.draw_normal(true);

    bbvi._optim = std::make_unique<RMSProp>(bbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                            bbvi.get_learning_rate(), 0.99);
    bbvi.draw_normal(false);
}

TEST_CASE("Draw variables", "[draw_variables]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);
    bbvi.draw_variables();
}

TEST_CASE("Get means and scales from q", "[get_means_and_scales_from_q]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi         = BBVI(neg_posterior, q, 3);
    auto means_scales = bbvi.get_means_and_scales_from_q();
    REQUIRE(means_scales.first == Eigen::Vector2d{q[0].get_mu0().value(), q[1].get_mu0().value()});
    REQUIRE(means_scales.second == Eigen::Vector2d{q[0].get_sigma0().value(), q[1].get_sigma0().value()});
}

TEST_CASE("Get means and scales", "[get_means_and_scales_from_q, get_means_and_scales]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    auto means_scales = bbvi.get_means_and_scales_from_q();
    REQUIRE(means_scales.first == Eigen::Vector2d{q[0].get_mu0().value(), q[1].get_mu0().value()});
    REQUIRE(means_scales.second == Eigen::Vector2d{q[0].get_sigma0().value(), q[1].get_sigma0().value()});

    bbvi._optim  = std::make_unique<RMSProp>(bbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                            bbvi.get_learning_rate(), 0.99);
    means_scales = bbvi.get_means_and_scales();
    REQUIRE(means_scales.first == bbvi._optim->get_parameters()(Eigen::seq(0, Eigen::last, 2)));
    REQUIRE(means_scales.second ==
            static_cast<Eigen::VectorXd>(bbvi._optim->get_parameters()(Eigen::seq(1, Eigen::last, 2)).array().exp()));
}

TEST_CASE("Compute the gradient of the approximating distributions", "[grad_log_q]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    int sims  = 3;
    BBVI bbvi = BBVI(neg_posterior, q, sims);

    Eigen::MatrixXd z        = Eigen::MatrixXd::Identity(2, sims);
    Eigen::Index param_count = 0;
    Eigen::MatrixXd grad     = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(bbvi.get_approx_param_no().sum()), sims);
    for (size_t core_param = 0; core_param < q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < q[core_param].get_param_no(); approx_param++) {
            Eigen::VectorXd temp_z = z.row(static_cast<Eigen::Index>(core_param));
            grad.row(param_count)  = q[core_param].vi_score(temp_z, approx_param);
            param_count++;
        }
    }

    REQUIRE(bbvi.grad_log_q(z) == grad);
}

TEST_CASE("Compute the unnormalized log posterior components", "[log_p]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[1]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    int sims  = 3;
    BBVI bbvi = BBVI(neg_posterior, q, sims);

    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, sims);
    REQUIRE(bbvi.log_p(z) == log_p_posterior(z, neg_posterior));
}

TEST_CASE("Compute the mean-field normal log posterior components", "[normal_log_q]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[1]; };
    std::vector<Normal> q{Normal(1.0, 1.5), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 3);

    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(2, 2);
    auto means_scales = bbvi.get_means_and_scales_from_q();
    REQUIRE(bbvi.normal_log_q(z, true) == Mvn::logpdf(z, means_scales.first, means_scales.second).rowwise().sum());

    bbvi._optim  = std::make_unique<RMSProp>(bbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                            bbvi.get_learning_rate(), 0.99);
    means_scales = bbvi.get_means_and_scales();
    REQUIRE(bbvi.normal_log_q(z, false) == Mvn::logpdf(z, means_scales.first, means_scales.second).rowwise().sum());
}

TEST_CASE("Print progress", "[print_progress]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[1]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi   = BBVI(neg_posterior, q, 3, "ADAM", 10);
    bbvi._optim = std::make_unique<ADAM>(bbvi.current_parameters(), Eigen::Vector4d::Zero(), bbvi.get_learning_rate(),
                                         0.9, 0.999);

    Eigen::VectorXd current_params = Eigen::Vector2d::Ones();
    bbvi.print_progress(2, current_params);
}

TEST_CASE("Get ELBO", "[get_elbo]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[1]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 2);

    bbvi._optim                    = std::make_unique<RMSProp>(bbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                            bbvi.get_learning_rate(), 0.99);
    Eigen::VectorXd current_params = Eigen::Vector2d::Ones();
    REQUIRE(bbvi.get_elbo(current_params) == -neg_posterior(current_params) - bbvi.create_normal_logq(current_params));
}

TEST_CASE("Run", "[run, run_with]") {
    std::function<double(Eigen::VectorXd)> neg_posterior = [](const Eigen::VectorXd& v) { return v[1]; };
    std::vector<Normal> q{Normal(1.0, 1.5), Normal(2.0, 2.5)};
    BBVI bbvi = BBVI(neg_posterior, q, 2, "ADAM", 5, 0.01, true, false);

    Eigen::MatrixXd z                = bbvi.draw_normal(true);
    Eigen::VectorXd gradient         = bbvi.cv_gradient(z, true);
    Eigen::VectorXd variance         = gradient.array().pow(2);
    Eigen::VectorXd final_parameters = bbvi.current_parameters();
    size_t final_samples             = 1;
    bbvi._optim = std::make_unique<ADAM>(final_parameters, variance, bbvi.get_learning_rate(), 0.9, 0.999);
    // bbvi._optim    = std::make_unique<RMSProp>(final_parameters, variance, bbvi.get_learning_rate(), 0.99);
    int iterations                               = bbvi.get_iterations();
    Eigen::MatrixXd stored_means                 = Eigen::MatrixXd::Zero(iterations, final_parameters.size() / 2);
    Eigen::VectorXd stored_predictive_likelihood = Eigen::VectorXd::Zero(iterations);
    Eigen::VectorXd elbo_records                 = Eigen::VectorXd::Zero(iterations);

    for (Eigen::Index i = 0; i < iterations; i++) {
        Eigen::MatrixXd x = bbvi.draw_normal();
        gradient          = bbvi.cv_gradient(x, false);
        Eigen::VectorXd optim_parameters{bbvi._optim->update(gradient)};
        bbvi.change_parameters(optim_parameters);
        optim_parameters                = bbvi._optim->get_parameters()(Eigen::seq(0, 1));
        stored_means.row(i)             = optim_parameters;
        stored_predictive_likelihood[i] = neg_posterior(stored_means.row(i));
        bbvi.print_progress(static_cast<double>(i), optim_parameters);
        if (static_cast<double>(i) > iterations - round(iterations / 10)) {
            final_samples++;
            final_parameters = final_parameters + bbvi._optim->get_parameters();
        }
        Eigen::VectorXd parameters = bbvi._optim->get_parameters()(Eigen::seq(0, 1));
        elbo_records[i]            = bbvi.get_elbo(parameters);
    }

    final_parameters = final_parameters / static_cast<double>(final_samples);
    bbvi.change_parameters(final_parameters);
    std::vector<double> means, ses;
    for (Eigen::Index i = 0; i < final_parameters.size(); i++) {
        if (i % 2 == 0)
            means.push_back(final_parameters[i]);
        else
            ses.push_back(final_parameters[i]);
    }
    Eigen::VectorXd final_means = Eigen::VectorXd::Map(means.data(), static_cast<Eigen::Index>(means.size()));
    Eigen::VectorXd final_ses   = Eigen::VectorXd::Map(ses.data(), static_cast<Eigen::Index>(ses.size()));

    BBVIReturnData result = bbvi.run(true);
    REQUIRE(result.q == q);
    REQUIRE(result.final_means == final_means);
    REQUIRE(result.final_ses == final_ses);
    REQUIRE(result.elbo_records == elbo_records);
    REQUIRE(result.stored_means == stored_means);
    REQUIRE(result.stored_predictive_likelihood == stored_predictive_likelihood);
    result = bbvi.run(false);
}

TEST_CASE("Create a CBBVI object", "[CBBVI]") {
    std::function<double(Eigen::VectorXd)> neg_posterior          = [](const Eigen::VectorXd& v) { return v[0]; };
    std::function<Eigen::VectorXd(Eigen::VectorXd)> log_p_blanket = [](const Eigen::VectorXd& v) { return v; };
    std::vector<Normal> q                                         = {Normal()};
    CBBVI cbbvi1 = CBBVI(neg_posterior, log_p_blanket, q, 3, "ADAM", 100, 0.01, false, false);
    CBBVI cbbvi2{cbbvi1};
    REQUIRE(cbbvi2 == cbbvi1);
    CBBVI cbbvi3{std::move(cbbvi1)};
    REQUIRE(cbbvi3 == cbbvi2);
    CBBVI cbbvi4 = CBBVI(neg_posterior, log_p_blanket, q, 0);
    cbbvi4       = cbbvi2;
    cbbvi4       = cbbvi3;
    REQUIRE(cbbvi4 == cbbvi2);
    cbbvi4 = std::move(cbbvi2);
    REQUIRE(cbbvi4 == cbbvi3);
}

TEST_CASE("Compute the unnormalized log posterior components (for CBBVI)", "[log_p]") {
    std::function<double(Eigen::VectorXd)> neg_posterior          = [](const Eigen::VectorXd& v) { return v[0]; };
    std::function<Eigen::VectorXd(Eigen::VectorXd)> log_p_blanket = [](const Eigen::VectorXd& v) { return v; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    int sims    = 3;
    CBBVI cbbvi = CBBVI(neg_posterior, log_p_blanket, q, sims);

    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, sims);
    Eigen::MatrixXd result(2, sims);
    for (Eigen::Index i = 0; i < 2; i++)
        result.row(i) = log_p_blanket(static_cast<Eigen::VectorXd>(z.row(i)));
    REQUIRE(cbbvi.log_p(z) == result);
}

TEST_CASE("Compute the mean-field normal log posterior components (for CBBVI)", "[normal_log_q]") {
    std::function<double(Eigen::VectorXd)> neg_posterior          = [](const Eigen::VectorXd& v) { return v[0]; };
    std::function<Eigen::VectorXd(Eigen::VectorXd)> log_p_blanket = [](const Eigen::VectorXd& v) { return v; };
    std::vector<Normal> q{Normal(1.0, 1.5), Normal(2.0, 2.5)};
    CBBVI cbbvi = CBBVI(neg_posterior, log_p_blanket, q, 3);

    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(2, 2);
    auto means_scales = cbbvi.get_means_and_scales_from_q();
    REQUIRE(cbbvi.normal_log_q(z, true) == Mvn::logpdf(z, means_scales.first, means_scales.second));

    cbbvi._optim = std::make_unique<RMSProp>(cbbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                             cbbvi.get_learning_rate(), 0.99);
    means_scales = cbbvi.get_means_and_scales();
    REQUIRE(cbbvi.normal_log_q(z, false) == Mvn::logpdf(z, means_scales.first, means_scales.second));
}

TEST_CASE("Compute cv_gradient (for CBBVI)", "[cv_gradient]") {
    std::function<double(Eigen::VectorXd)> neg_posterior          = [](const Eigen::VectorXd& v) { return v[0]; };
    std::function<Eigen::VectorXd(Eigen::VectorXd)> log_p_blanket = [](const Eigen::VectorXd& v) { return v; };

    std::vector<Normal> q = std::vector<Normal>{Normal(), Normal()};
    int sims              = 3;
    CBBVI cbbvi           = CBBVI(neg_posterior, log_p_blanket, q, sims);

    Eigen::MatrixXd z          = Eigen::MatrixXd::Identity(2, sims);
    Eigen::MatrixXd z_t        = z.transpose();
    Eigen::MatrixXd log_q      = cbbvi.normal_log_q(z_t, true);
    log_q                      = log_q.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
    Eigen::MatrixXd log_p      = cbbvi.log_p(z_t);
    Eigen::MatrixXd grad_log_q = cbbvi.grad_log_q(z);
    Eigen::MatrixXd sub_log(4, sims);
    sub_log << (log_p - log_q).transpose(), (log_p - log_q).transpose();
    Eigen::MatrixXd gradient = grad_log_q.array() * sub_log.array();

    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(cbbvi.get_approx_param_no().sum()));
    alpha_recursion(alpha0, grad_log_q, gradient, static_cast<size_t>(cbbvi.get_approx_param_no().sum()));
    double var = pow((grad_log_q.array() - grad_log_q.mean()).abs(), 2).mean();
    Eigen::MatrixXd sub(gradient.cols(), gradient.rows());
    for (Eigen::Index i = 0; i < sub.rows(); i++)
        sub.row(i) = (alpha0.transpose().array() / var) * grad_log_q.transpose().row(i).array();
    Eigen::MatrixXd vectorized = gradient - sub.transpose();

    REQUIRE(cbbvi.cv_gradient(z, true) == vectorized.rowwise().mean());

    cbbvi._optim = std::make_unique<RMSProp>(cbbvi.current_parameters(), Eigen::Vector4d::Zero(),
                                             cbbvi.get_learning_rate(), 0.99);
    log_q        = cbbvi.normal_log_q(z_t, false);
    sub_log << (log_p - log_q).transpose(), (log_p - log_q).transpose();
    gradient = grad_log_q.array() * sub_log.array();
    alpha_recursion(alpha0, grad_log_q, gradient, static_cast<size_t>(cbbvi.get_approx_param_no().sum()));
    var = pow((grad_log_q.array() - grad_log_q.mean()).abs(), 2).mean();
    for (Eigen::Index i = 0; i < sub.rows(); i++)
        sub.row(i) = (alpha0.transpose().array() / var) * grad_log_q.transpose().row(i).array();
    vectorized = gradient - sub.transpose();
    REQUIRE(cbbvi.cv_gradient(z, false) == vectorized.rowwise().mean());
}

TEST_CASE("Create a BBVIM object", "[BBVIM]") {
    std::function<double(Eigen::VectorXd, int)> neg_posterior = [](const Eigen::VectorXd& v, int n) { return v[n]; };
    std::function<double(Eigen::VectorXd)> full_neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q                                     = {Normal()};
    BBVIM bbvim1 = BBVIM(neg_posterior, full_neg_posterior, q, 3, "ADAM", 100, 0.01, 1, false, false);
    BBVIM bbvim2{bbvim1};
    REQUIRE(bbvim2 == bbvim1);
    BBVIM bbvim3{std::move(bbvim1)};
    REQUIRE(bbvim3 == bbvim2);
    BBVIM bbvim4 = BBVIM(neg_posterior, full_neg_posterior, q, 0);
    bbvim4       = bbvim2;
    bbvim4       = bbvim3;
    REQUIRE(bbvim4 == bbvim2);
    bbvim4 = std::move(bbvim2);
    REQUIRE(bbvim4 == bbvim3);
}

TEST_CASE("Compute the unnormalized log posterior components (for BVVIM)", "[log_p]") {
    std::function<double(Eigen::VectorXd, int)> neg_posterior = [](const Eigen::VectorXd& v, int n) { return v[n]; };
    std::function<double(Eigen::VectorXd)> full_neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    int sims    = 3;
    BBVIM bbvim = BBVIM(neg_posterior, full_neg_posterior, q, sims);

    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(2, sims);
    REQUIRE(bbvim.log_p(z) == mb_log_p_posterior(z, neg_posterior, 2));
}

TEST_CASE("Get ELBO (for BBVIM)", "[get_elbo]") {
    std::function<double(Eigen::VectorXd, int)> neg_posterior = [](const Eigen::VectorXd& v, int n) { return v[n]; };
    std::function<double(Eigen::VectorXd)> full_neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVIM bbvim = BBVIM(neg_posterior, full_neg_posterior, q, 2);

    bbvim._optim                   = std::make_unique<RMSProp>(bbvim.current_parameters(), Eigen::Vector4d::Zero(),
                                             bbvim.get_learning_rate(), 0.99);
    Eigen::VectorXd current_params = Eigen::Vector2d::Ones();
    REQUIRE(bbvim.get_elbo(current_params) ==
            full_neg_posterior(current_params) - bbvim.create_normal_logq(current_params));
}

TEST_CASE("Print progress (for BBVIM)", "[print_progress]") {
    std::function<double(Eigen::VectorXd, int)> neg_posterior = [](const Eigen::VectorXd& v, int n) { return v[n]; };
    std::function<double(Eigen::VectorXd)> full_neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVIM bbvim = BBVIM(neg_posterior, full_neg_posterior, q, 2, "ADAM", 10);

    bbvim._optim = std::make_unique<ADAM>(bbvim.current_parameters(), Eigen::Vector4d::Zero(),
                                          bbvim.get_learning_rate(), 0.9, 0.999);

    Eigen::VectorXd current_params = Eigen::Vector2d::Ones();
    bbvim.print_progress(2, current_params);
}

TEST_CASE("Run (for BBVIM)", "[run, run_with]") {
    std::function<double(Eigen::VectorXd, int)> neg_posterior = [](const Eigen::VectorXd& v, int n) { return v[0]; };
    std::function<double(Eigen::VectorXd)> full_neg_posterior = [](const Eigen::VectorXd& v) { return v[0]; };
    std::vector<Normal> q{Normal(), Normal(2.0, 2.5)};
    BBVIM bbvim = BBVIM(neg_posterior, full_neg_posterior, q, 2, "ADAM", 5, 0.01, 2, true, false);

    Eigen::MatrixXd z                = bbvim.draw_normal(true);
    Eigen::VectorXd gradient         = bbvim.cv_gradient(z, true);
    Eigen::VectorXd variance         = gradient.array().pow(2);
    Eigen::VectorXd final_parameters = bbvim.current_parameters();
    size_t final_samples             = 1;
    bbvim._optim = std::make_unique<ADAM>(final_parameters, variance, bbvim.get_learning_rate(), 0.9, 0.999);
    // bbvim._optim    = std::make_unique<RMSProp>(final_parameters, variance, bbvim.get_learning_rate(), 0.99);
    int iterations                               = bbvim.get_iterations();
    Eigen::MatrixXd stored_means                 = Eigen::MatrixXd::Zero(iterations, final_parameters.size() / 2);
    Eigen::VectorXd stored_predictive_likelihood = Eigen::VectorXd::Zero(iterations);
    Eigen::VectorXd elbo_records                 = Eigen::VectorXd::Zero(iterations);

    for (Eigen::Index i = 0; i < iterations; i++) {
        Eigen::MatrixXd x = bbvim.draw_normal();
        gradient          = bbvim.cv_gradient(x, false);
        Eigen::VectorXd optim_parameters{bbvim._optim->update(gradient)};
        bbvim.change_parameters(optim_parameters);
        optim_parameters                = bbvim._optim->get_parameters()(Eigen::seq(0, 1));
        stored_means.row(i)             = optim_parameters;
        stored_predictive_likelihood[i] = full_neg_posterior(stored_means.row(i));
        bbvim.print_progress(static_cast<double>(i), optim_parameters);
        if (static_cast<double>(i) > iterations - round(iterations / 10)) {
            final_samples++;
            final_parameters = final_parameters + bbvim._optim->get_parameters();
        }
        Eigen::VectorXd parameters = bbvim._optim->get_parameters()(Eigen::seq(0, 1));
        elbo_records[i]            = bbvim.get_elbo(parameters);
    }

    final_parameters = final_parameters / static_cast<double>(final_samples);
    bbvim.change_parameters(final_parameters);
    std::vector<double> means, ses;
    for (Eigen::Index i = 0; i < final_parameters.size(); i++) {
        if (i % 2 == 0)
            means.push_back(final_parameters[i]);
        else
            ses.push_back(final_parameters[i]);
    }
    Eigen::VectorXd final_means = Eigen::VectorXd::Map(means.data(), static_cast<Eigen::Index>(means.size()));
    Eigen::VectorXd final_ses   = Eigen::VectorXd::Map(ses.data(), static_cast<Eigen::Index>(ses.size()));

    BBVIReturnData result = bbvim.run(true);
    REQUIRE(result.q == q);
    REQUIRE(result.final_means == final_means);
    REQUIRE(result.final_ses == final_ses);
    REQUIRE(result.elbo_records == elbo_records);
    REQUIRE(result.stored_means == stored_means);
    REQUIRE(result.stored_predictive_likelihood == stored_predictive_likelihood);
    result = bbvim.run(false);
}