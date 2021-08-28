#include "inference/bbvi.hpp"

#include "inference/bbvi_routines.hpp"
#include "multivariate_normal.hpp"

BBVI::BBVI(std::function<double(Eigen::VectorXd)> neg_posterior, std::vector<Normal>& q, int sims,
           std::string optimizer, int iterations, double learning_rate, bool record_elbo, bool quiet_progress)
    : _neg_posterior{neg_posterior}, _q{q}, _sims{sims}, _printer{true}, _optimizer{optimizer}, _iterations{iterations},
      _learning_rate{learning_rate}, _record_elbo{record_elbo}, _quiet_progress{quiet_progress} {
    _approx_param_no = Eigen::VectorXd(_q.size());
    for (Eigen::Index i{0}; i < _q.size(); i++) {
        _approx_param_no[i] = _q[i].get_param_no();
    }
}

BBVI::~BBVI() = default;

void BBVI::change_parameters(Eigen::VectorXd& params) {
    Eigen::Index no_of_params = 0;
    for (auto& normal : _q) {
        for (size_t approx_param = 0; approx_param < normal.get_param_no(); approx_param++) {
            normal.vi_change_param(approx_param, params[no_of_params]);
            no_of_params++;
        }
    }
}

double BBVI::create_normal_logq(Eigen::VectorXd& z) {
    auto means_scales = BBVI::get_means_and_scales();
    return Mvn::logpdf(z, means_scales.first, means_scales.second).sum();
}

Eigen::VectorXd BBVI::cv_gradient(Eigen::MatrixXd& z, bool initial) {
    Eigen::VectorXd gradient;
    Eigen::MatrixXd z_t            = z.transpose();
    Eigen::VectorXd log_q_res      = normal_log_q(z_t, initial);
    Eigen::VectorXd log_p_res      = log_p(z_t);
    Eigen::MatrixXd grad_log_q_res = grad_log_q(z);
    gradient                       = grad_log_q_res * (log_p_res - log_q_res);

    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_approx_param_no.sum()));
    alpha_recursion(alpha0, grad_log_q_res, gradient, static_cast<size_t>(_approx_param_no.sum()));

    double var                 = pow((grad_log_q_res.array() - grad_log_q_res.mean()).abs(), 2).mean();
    Eigen::VectorXd vectorized = gradient - ((alpha0 / var) * grad_log_q_res.transpose()).transpose();

    return vectorized.colwise().mean();
}

Eigen::VectorXd BBVI::current_parameters() {
    std::vector<double> current = std::vector<double>();
    for (auto& normal : _q) {
        for (size_t approx_param = 0; approx_param < normal.get_param_no(); approx_param++)
            current.push_back(normal.vi_return_param(approx_param));
    }
    return Eigen::VectorXd::Map(current.data(), static_cast<Eigen::Index>(current.size()));
}

Eigen::MatrixXd BBVI::draw_normal(bool initial) {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> pair;
    if (initial)
        pair = get_means_and_scales_from_q();
    else
        pair = get_means_and_scales();

    Eigen::MatrixXd normal = Eigen::MatrixXd(_sims, pair.first.size());
    for (Eigen::Index i = 0; i < _sims; i++)
        normal.row(i) = Mvn::random(pair.first[i], pair.second[i], pair.first.size());
    return normal.transpose();
}

Eigen::MatrixXd BBVI::draw_variables() {
    Eigen::MatrixXd z = Eigen::MatrixXd(_q.size(), _sims);
    for (Eigen::Index i = 0; i < _q.size(); i++)
        z.row(i) = _q[i].draw_variable_local(_sims);
    return z;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> BBVI::get_means_and_scales_from_q() {
    Eigen::VectorXd means = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_q.size()));
    Eigen::VectorXd scale = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_q.size()));

    for (Eigen::Index i = 0; i < _q.size(); i++) {
        means(i) = _q[i].vi_return_param(0);
        scale(i) = _q[i].vi_return_param(1);
    }

    return {means, scale};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> BBVI::get_means_and_scales() {
    return {_optim->get_parameters()(Eigen::seq(0, Eigen::last, 2)),
            _optim->get_parameters()(Eigen::seq(1, Eigen::last, 2))};
}


Eigen::MatrixXd BBVI::grad_log_q(Eigen::MatrixXd& z) {
    Eigen::Index param_count = 0;
    Eigen::MatrixXd grad     = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(_approx_param_no.sum()), _sims);
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++) {
            Eigen::VectorXd temp_z = z.row(static_cast<Eigen::Index>(core_param));
            grad.row(param_count)  = _q[core_param].vi_score(temp_z, approx_param);
            param_count++;
        }
    }
    return grad;
}

Eigen::VectorXd BBVI::log_p(Eigen::MatrixXd& z) {
    return log_p_posterior(z, _neg_posterior);
}

std::vector<Normal> BBVI::get_q() const {
    return _q;
}

Eigen::VectorXd BBVI::normal_log_q(Eigen::MatrixXd& z, bool initial) {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> means_scales;
    if (initial)
        means_scales = get_means_and_scales_from_q();
    else
        means_scales = get_means_and_scales();
    return Mvn::logpdf(z, means_scales.first, means_scales.second).colwise().sum();
}

void BBVI::print_progress(double i, Eigen::VectorXd& current_params) {
    for (int split{1}; split < 11; split++) {
        if (i == round(_iterations / 10 * split) - 1) {
            double post   = -_neg_posterior(current_params);
            double approx = create_normal_logq(current_params);
            double diff   = post - approx;
            if (!_quiet_progress) {
                std::cout << split << "0% done : ELBO is " << diff << ", p(y,z) is " << post << ", q(z) is " << approx;
            }
        }
    }
}

double BBVI::get_elbo(Eigen::VectorXd& current_params) {
    return -_neg_posterior(current_params) - create_normal_logq(current_params);
}

BBVIReturnData BBVI::run() {
    // Initialization assumptions
    Eigen::MatrixXd z        = draw_normal(true);
    Eigen::VectorXd gradient = cv_gradient(z, true);
    for (Eigen::Index i = 0; i < gradient.size(); i++) {
        if (std::isnan(gradient[i]))
            gradient[i] = 0;
    }
    Eigen::VectorXd variance         = gradient.array().pow(2);
    Eigen::VectorXd final_parameters = current_parameters();
    size_t final_samples             = 1;

    // Create optimizer
    if (_optimizer == "ADAM")
        _optim.reset(new ADAM(final_parameters, variance, _learning_rate, 0.9, 0.999));
    else if (_optimizer == "RMSProp")
        _optim.reset(new RMSProp(final_parameters, variance, _learning_rate, 0.99));

    // Record elbo
    Eigen::VectorXd elbo_records;
    if (_record_elbo)
        elbo_records = Eigen::VectorXd::Zero(_iterations);

    for (Eigen::Index i = 0; i < _iterations; i++) {
        Eigen::MatrixXd x = draw_normal();
        gradient          = cv_gradient(x, false);
        for (Eigen::Index j = 0; j < gradient.size(); j++) {
            if (std::isnan(gradient[j]))
                gradient[j] = 0;
        }
        Eigen::VectorXd optim_parameters{_optim->update(gradient)};
        change_parameters(optim_parameters);

        optim_parameters = _optim->get_parameters()(Eigen::seq(0, 2));
        if (_printer)
            print_progress(static_cast<double>(i), optim_parameters);

        // Construct final parameters using final 10% of samples
        if (static_cast<double>(i) > _iterations - round(_iterations / 10)) {
            final_samples++;
            final_parameters = final_parameters + _optim->get_parameters();
        }

        if (_record_elbo) {
            Eigen::VectorXd parameters = _optim->get_parameters()(Eigen::seq(0, 2));
            elbo_records[i]            = get_elbo(parameters);
        }
    }

    final_parameters = final_parameters / static_cast<double>(final_samples);
    change_parameters(final_parameters);

    std::vector<double> means, ses;
    for (Eigen::Index i = 0; i < final_parameters.size(); i++) {
        if (i % 2 == 0)
            means.push_back(final_parameters[i]);
        else
            ses.push_back(final_parameters[i]);
    }
    Eigen::VectorXd final_means = Eigen::VectorXd::Map(means.data(), static_cast<Eigen::Index>(means.size()));
    Eigen::VectorXd final_ses   = Eigen::VectorXd::Map(ses.data(), static_cast<Eigen::Index>(ses.size()));

    if (!_quiet_progress)
        std::cout << "\nFinal model ELBO is " << -_neg_posterior(final_means) - create_normal_logq(final_means) << "\n";

    return {_q, final_means, final_ses, elbo_records};
}

BBVIReturnData BBVI::run_and_store() {
    // Initialization assumptions
    Eigen::MatrixXd z        = draw_normal(true);
    Eigen::VectorXd gradient = cv_gradient(z, true);
    for (Eigen::Index i = 0; i < gradient.size(); i++) {
        if (std::isnan(gradient[i]))
            gradient[i] = 0;
    }
    Eigen::VectorXd variance         = gradient.array().pow(2);
    Eigen::VectorXd final_parameters = current_parameters();
    size_t final_samples             = 1;

    // Create optimizer
    if (_optimizer == "ADAM")
        _optim.reset(new ADAM(final_parameters, variance, _learning_rate, 0.9, 0.999));
    else if (_optimizer == "RMSProp")
        _optim.reset(new RMSProp(final_parameters, variance, _learning_rate, 0.99));

    // Store updates
    Eigen::MatrixXd stored_means                 = Eigen::MatrixXd::Zero(_iterations, final_parameters.size() / 2);
    Eigen::VectorXd stored_predictive_likelihood = Eigen::VectorXd::Zero(_iterations);

    // Record elbo
    Eigen::VectorXd elbo_records;
    if (_record_elbo)
        elbo_records = Eigen::VectorXd::Zero(_iterations);

    for (Eigen::Index i = 0; i < _iterations; i++) {
        Eigen::MatrixXd x = draw_normal();
        gradient          = cv_gradient(x, false);
        for (Eigen::Index j = 0; j < gradient.size(); j++) {
            if (std::isnan(gradient[j]))
                gradient[j] = 0;
        }
        Eigen::VectorXd optim_parameters{_optim->update(gradient)};
        change_parameters(optim_parameters);

        optim_parameters                = _optim->get_parameters()(Eigen::seq(0, 2));
        stored_means.row(i)             = optim_parameters;
        stored_predictive_likelihood[i] = _neg_posterior(stored_means.row(i));

        if (_printer)
            print_progress(static_cast<double>(i), optim_parameters);

        // Construct final parameters using final 10% of samples
        if (static_cast<double>(i) > _iterations - round(_iterations / 10)) {
            final_samples++;
            final_parameters = final_parameters + _optim->get_parameters();
        }

        if (_record_elbo) {
            Eigen::VectorXd parameters = _optim->get_parameters()(Eigen::seq(0, 2));
            elbo_records[i]            = get_elbo(parameters);
        }
    }

    final_parameters = final_parameters / static_cast<double>(final_samples);
    change_parameters(final_parameters);

    std::vector<double> means, ses;
    for (Eigen::Index i = 0; i < final_parameters.size(); i++) {
        if (i % 2 == 0)
            means.push_back(final_parameters[i]);
        else
            ses.push_back(final_parameters[i]);
    }
    Eigen::VectorXd final_means = Eigen::VectorXd::Map(means.data(), static_cast<Eigen::Index>(means.size()));
    Eigen::VectorXd final_ses   = Eigen::VectorXd::Map(ses.data(), static_cast<Eigen::Index>(ses.size()));

    if (!_quiet_progress)
        std::cout << "\nFinal model ELBO is " << -_neg_posterior(final_means) - create_normal_logq(final_means) << "\n";

    return {_q, final_means, final_ses, elbo_records};
}

CBBVI::CBBVI(std::function<double(Eigen::VectorXd)> neg_posterior, std::function<double(Eigen::VectorXd)> log_p_blanket,
             std::vector<Normal>& q, int sims, std::string optimizer, int iterations, double learning_rate,
             bool record_elbo, bool quiet_progress)
    : BBVI{neg_posterior, q, sims, optimizer, iterations, learning_rate, record_elbo, quiet_progress},
      _log_p_blanket{log_p_blanket} {}

Eigen::VectorXd CBBVI::log_p(Eigen::MatrixXd& z) {
    std::vector<double> result;
    for (Eigen::Index i = 0; i < z.size(); i++)
        result.push_back(_log_p_blanket(static_cast<Eigen::VectorXd>(z.row(i))));
    return Eigen::VectorXd::Map(result.data(), static_cast<Eigen::Index>(result.size()));
}

Eigen::VectorXd CBBVI::normal_log_q(Eigen::MatrixXd& z, bool initial) {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> means_scales;
    if (initial)
        means_scales = get_means_and_scales_from_q();
    else
        means_scales = get_means_and_scales();
    return Mvn::logpdf(z, means_scales.first, means_scales.second);
}

Eigen::VectorXd CBBVI::cv_gradient(Eigen::MatrixXd& z, bool initial) {
    Eigen::VectorXd gradient;
    Eigen::MatrixXd z_t            = z.transpose();
    Eigen::VectorXd log_q_res      = normal_log_q(z_t, initial);
    Eigen::VectorXd log_p_res      = log_p(z_t);
    Eigen::MatrixXd grad_log_q_res = grad_log_q(z);
    Eigen::MatrixXd sub_log;
    sub_log << (log_p_res - log_q_res).transpose(), (log_p_res - log_q_res).transpose();
    gradient = grad_log_q_res * sub_log;

    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_approx_param_no.sum()));
    alpha_recursion(alpha0, grad_log_q_res, gradient, static_cast<size_t>(_approx_param_no.sum()));

    double var                 = pow((grad_log_q_res.array() - grad_log_q_res.mean()).abs(), 2).mean();
    Eigen::VectorXd vectorized = gradient - ((alpha0 / var) * grad_log_q_res.transpose()).transpose();

    return vectorized.colwise().mean();
}

BBVIM::BBVIM(std::function<double(Eigen::VectorXd, int)> neg_posterior,
             std::function<double(Eigen::VectorXd)> full_neg_posterior, std::vector<Normal>& q, int sims,
             std::string optimizer, int iterations, double learning_rate, int mini_batch, bool record_elbo,
             bool quiet_progress)
    : BBVI{std::function<double(Eigen::VectorXd)>(),
           q,
           sims,
           optimizer,
           iterations,
           learning_rate,
           record_elbo,
           quiet_progress},
      _neg_posterior{neg_posterior}, _full_neg_posterior{full_neg_posterior}, _mini_batch{mini_batch} {}

Eigen::VectorXd BBVIM::log_p(Eigen::MatrixXd& z) {
    return mb_log_p_posterior(z, _neg_posterior, _mini_batch);
}

double BBVIM::get_elbo(Eigen::VectorXd& current_params) {
    return _full_neg_posterior(current_params) - create_normal_logq(current_params);
}

void BBVIM::print_progress(double i, Eigen::VectorXd& current_params) {
    for (int split{1}; split < 11; split++) {
        if (i == round(_iterations / 10 * split) - 1) {
            double post   = -_full_neg_posterior(current_params);
            double approx = create_normal_logq(current_params);
            double diff   = post - approx;
            if (!_quiet_progress) {
                std::cout << split << "0% done : ELBO is " << diff << ", p(y,z) is " << post << ", q(z) is " << approx;
            }
        }
    }
}

BBVIReturnData BBVIM::run() {
    // Initialization assumptions
    Eigen::MatrixXd z        = draw_normal(true);
    Eigen::VectorXd gradient = cv_gradient(z, true);
    for (Eigen::Index i = 0; i < gradient.size(); i++) {
        if (std::isnan(gradient[i]))
            gradient[i] = 0;
    }
    Eigen::VectorXd variance         = gradient.array().pow(2);
    Eigen::VectorXd final_parameters = current_parameters();
    size_t final_samples             = 1;

    // Create optimizer
    if (_optimizer == "ADAM")
        _optim.reset(new ADAM(final_parameters, variance, _learning_rate, 0.9, 0.999));
    else if (_optimizer == "RMSProp")
        _optim.reset(new RMSProp(final_parameters, variance, _learning_rate, 0.99));

    // Record elbo
    Eigen::VectorXd elbo_records;
    if (_record_elbo)
        elbo_records = Eigen::VectorXd::Zero(_iterations);

    for (Eigen::Index i = 0; i < _iterations; i++) {
        Eigen::MatrixXd x = draw_normal();
        gradient          = cv_gradient(x, false);
        for (Eigen::Index j = 0; j < gradient.size(); j++) {
            if (std::isnan(gradient[j]))
                gradient[j] = 0;
        }
        Eigen::VectorXd optim_parameters{_optim->update(gradient)};
        change_parameters(optim_parameters);

        optim_parameters = _optim->get_parameters()(Eigen::seq(0, 2));
        if (_printer)
            print_progress(static_cast<double>(i), optim_parameters);

        // Construct final parameters using final 10% of samples
        if (static_cast<double>(i) > _iterations - round(_iterations / 10)) {
            final_samples++;
            final_parameters = final_parameters + _optim->get_parameters();
        }

        if (_record_elbo) {
            Eigen::VectorXd parameters = _optim->get_parameters()(Eigen::seq(0, 2));
            elbo_records[i]            = get_elbo(parameters);
        }
    }

    final_parameters = final_parameters / static_cast<double>(final_samples);
    change_parameters(final_parameters);

    std::vector<double> means, ses;
    for (Eigen::Index i = 0; i < final_parameters.size(); i++) {
        if (i % 2 == 0)
            means.push_back(final_parameters[i]);
        else
            ses.push_back(final_parameters[i]);
    }
    Eigen::VectorXd final_means = Eigen::VectorXd::Map(means.data(), static_cast<Eigen::Index>(means.size()));
    Eigen::VectorXd final_ses   = Eigen::VectorXd::Map(ses.data(), static_cast<Eigen::Index>(ses.size()));

    if (!_quiet_progress)
        std::cout << "\nFinal model ELBO is " << -_full_neg_posterior(final_means) - create_normal_logq(final_means)
                  << "\n";

    return {_q, final_means, final_ses, elbo_records};
}

BBVIReturnData BBVIM::run_and_store() {
    // Initialization assumptions
    Eigen::MatrixXd z        = draw_normal(true);
    Eigen::VectorXd gradient = cv_gradient(z, true);
    for (Eigen::Index i = 0; i < gradient.size(); i++) {
        if (std::isnan(gradient[i]))
            gradient[i] = 0;
    }
    Eigen::VectorXd variance         = gradient.array().pow(2);
    Eigen::VectorXd final_parameters = current_parameters();
    size_t final_samples             = 1;

    // Create optimizer
    if (_optimizer == "ADAM")
        _optim.reset(new ADAM(final_parameters, variance, _learning_rate, 0.9, 0.999));
    else if (_optimizer == "RMSProp")
        _optim.reset(new RMSProp(final_parameters, variance, _learning_rate, 0.99));

    // Store updates
    Eigen::MatrixXd stored_means                 = Eigen::MatrixXd::Zero(_iterations, final_parameters.size() / 2);
    Eigen::VectorXd stored_predictive_likelihood = Eigen::VectorXd::Zero(_iterations);

    // Record elbo
    Eigen::VectorXd elbo_records;
    if (_record_elbo)
        elbo_records = Eigen::VectorXd::Zero(_iterations);

    for (Eigen::Index i = 0; i < _iterations; i++) {
        Eigen::MatrixXd x = draw_normal();
        gradient          = cv_gradient(x, false);
        for (Eigen::Index j = 0; j < gradient.size(); j++) {
            if (std::isnan(gradient[j]))
                gradient[j] = 0;
        }
        Eigen::VectorXd optim_parameters{_optim->update(gradient)};
        change_parameters(optim_parameters);

        optim_parameters                = _optim->get_parameters()(Eigen::seq(0, 2));
        stored_means.row(i)             = optim_parameters;
        stored_predictive_likelihood[i] = _neg_posterior(stored_means.row(i), _mini_batch);
        //@FIXME: nell'originale richiama _neg_posterior con un solo parametro

        if (_printer)
            print_progress(static_cast<double>(i), optim_parameters);

        // Construct final parameters using final 10% of samples
        if (static_cast<double>(i) > _iterations - round(_iterations / 10)) {
            final_samples++;
            final_parameters = final_parameters + _optim->get_parameters();
        }

        if (_record_elbo) {
            Eigen::VectorXd parameters = _optim->get_parameters()(Eigen::seq(0, 2));
            elbo_records[i]            = get_elbo(parameters);
        }
    }

    final_parameters = final_parameters / static_cast<double>(final_samples);
    change_parameters(final_parameters);

    std::vector<double> means, ses;
    for (Eigen::Index i = 0; i < final_parameters.size(); i++) {
        if (i % 2 == 0)
            means.push_back(final_parameters[i]);
        else
            ses.push_back(final_parameters[i]);
    }
    Eigen::VectorXd final_means = Eigen::VectorXd::Map(means.data(), static_cast<Eigen::Index>(means.size()));
    Eigen::VectorXd final_ses   = Eigen::VectorXd::Map(ses.data(), static_cast<Eigen::Index>(ses.size()));

    if (!_quiet_progress)
        std::cout << "\nFinal model ELBO is " << -_full_neg_posterior(final_means) - create_normal_logq(final_means)
                  << "\n";

    return {_q, final_means, final_ses, elbo_records};
}