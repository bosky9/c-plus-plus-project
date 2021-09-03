#include "inference/bbvi.hpp"

#include <memory>
#include <utility>

#include "inference/bbvi_routines.hpp"
#include "multivariate_normal.hpp"

BBVI::BBVI(std::function<double(Eigen::VectorXd)> neg_posterior, const std::vector<Normal>& q, int sims,
           std::string optimizer, int iterations, double learning_rate, bool record_elbo, bool quiet_progress)
    : _neg_posterior{std::move(neg_posterior)}, _q{q}, _sims{sims}, _printer{true}, _optimizer{std::move(optimizer)},
      _iterations{iterations}, _learning_rate{learning_rate}, _record_elbo{record_elbo}, _quiet_progress{
                                                                                                 quiet_progress} {
    _approx_param_no = Eigen::VectorXd(_q.size());
    for (Eigen::Index i{0}; i < _q.size(); i++) {
        _approx_param_no[i] = _q[i].get_param_no();
    }
}

BBVI::BBVI(const BBVI& bbvi)
    : _neg_posterior{bbvi._neg_posterior}, _q{bbvi._q}, _sims{bbvi._sims}, _approx_param_no{bbvi._approx_param_no},
      _printer{bbvi._printer}, _iterations{bbvi._iterations}, _record_elbo{bbvi._record_elbo},
      _quiet_progress{bbvi._quiet_progress}, _optimizer{bbvi._optimizer}, _learning_rate{bbvi._learning_rate} {
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(bbvi._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(bbvi._optim)));
    */
}

BBVI::BBVI(BBVI&& bbvi) noexcept : BBVI(bbvi) {
    bbvi._neg_posterior = {};
    bbvi._q.resize(0);
    bbvi._sims = 0;
    bbvi._approx_param_no.resize(0, 0);
    bbvi._printer        = false;
    bbvi._iterations     = 0;
    bbvi._record_elbo    = false;
    bbvi._quiet_progress = true;
    bbvi._optimizer      = nullptr;
    bbvi._learning_rate  = 0;
}

BBVI& BBVI::operator=(const BBVI& bbvi) {
    if (this == &bbvi)
        return *this;
    _neg_posterior   = bbvi._neg_posterior;
    _q               = bbvi._q;
    _sims            = bbvi._sims;
    _approx_param_no = bbvi._approx_param_no;
    _printer         = bbvi._printer;
    _iterations      = bbvi._iterations;
    _record_elbo     = bbvi._record_elbo;
    _quiet_progress  = bbvi._quiet_progress;
    _optimizer       = bbvi._optimizer;
    _learning_rate   = bbvi._learning_rate;
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(bbvi._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(bbvi._optim)));
    */
    return *this;
}

BBVI& BBVI::operator=(BBVI&& bbvi) noexcept {
    _neg_posterior   = bbvi._neg_posterior;
    _q               = bbvi._q;
    _sims            = bbvi._sims;
    _approx_param_no = bbvi._approx_param_no;
    _printer         = bbvi._printer;
    _iterations      = bbvi._iterations;
    _record_elbo     = bbvi._record_elbo;
    _quiet_progress  = bbvi._quiet_progress;
    _optimizer       = bbvi._optimizer;
    _learning_rate   = bbvi._learning_rate;
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(bbvi._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(bbvi._optim)));
    */
    bbvi._neg_posterior = {};
    bbvi._q.resize(0);
    bbvi._sims = 0;
    bbvi._approx_param_no.resize(0, 0);
    bbvi._printer        = false;
    bbvi._iterations     = 0;
    bbvi._record_elbo    = false;
    bbvi._quiet_progress = true;
    bbvi._optimizer      = nullptr;
    bbvi._learning_rate  = 0;
    return *this;
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
    assert(z.cols() == _sims);
    Eigen::MatrixXd z_t            = z.transpose();
    Eigen::VectorXd log_q_res      = normal_log_q(z_t, initial);
    Eigen::VectorXd log_p_res      = log_p(z_t);
    Eigen::MatrixXd grad_log_q_res = grad_log_q(z);
    gradient                       = grad_log_q_res * (log_p_res - log_q_res);

    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_approx_param_no.sum()));
    alpha_recursion(alpha0, grad_log_q_res, gradient, static_cast<size_t>(_approx_param_no.sum()));

    double var = pow((grad_log_q_res.array() - grad_log_q_res.mean()).abs(), 2).mean();
    Eigen::MatrixXd vectorized(gradient.rows(), gradient.cols());
    Eigen::MatrixXd sub(gradient.cols(), gradient.rows());
    std::cout << alpha0.size() << " " << grad_log_q_res.rows() << " " << grad_log_q_res.cols() << "\n";
    for (Eigen::Index i = 0; i < sub.rows(); i++)
        sub.row(i) = (alpha0.transpose().array() / var) * grad_log_q_res.transpose().row(i).array();
    vectorized = gradient - sub.transpose();

    return vectorized.rowwise().mean();
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
    return Mvn::logpdf(z, means_scales.first, means_scales.second).rowwise().sum();
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

BBVIReturnData BBVI::run_with(bool store, const std::function<double(Eigen::VectorXd)>& neg_posterior) {
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
        _optim = std::make_unique<ADAM>(final_parameters, variance, _learning_rate, 0.9, 0.999);
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(final_parameters, variance, _learning_rate, 0.99);

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
        optim_parameters = _optim->get_parameters()(Eigen::seq(0, 2));

        if (store) {
            stored_means.row(i)             = optim_parameters;
            stored_predictive_likelihood[i] = neg_posterior(stored_means.row(i));
        }

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
        std::cout << "\nFinal model ELBO is " << -neg_posterior(final_means) - create_normal_logq(final_means) << "\n";

    if (store) {
        return {_q, final_means, final_ses, elbo_records, stored_means, stored_predictive_likelihood};
    }
    return {_q, final_means, final_ses, elbo_records};
}

BBVIReturnData BBVI::run(bool store) {
    return run_with(store, _neg_posterior);
}

CBBVI::CBBVI(std::function<double(Eigen::VectorXd)> neg_posterior, std::function<double(Eigen::VectorXd)> log_p_blanket,
             std::vector<Normal>& q, int sims, std::string optimizer, int iterations, double learning_rate,
             bool record_elbo, bool quiet_progress)
    : BBVI{std::move(neg_posterior),
           q,
           sims,
           std::move(optimizer),
           iterations,
           learning_rate,
           record_elbo,
           quiet_progress},
      _log_p_blanket{std::move(log_p_blanket)} {}

CBBVI::CBBVI(const CBBVI& cbbvi)
    : _log_p_blanket{cbbvi._log_p_blanket}, BBVI{cbbvi._neg_posterior, cbbvi._q,
                                                 cbbvi._sims,          cbbvi._optimizer,
                                                 cbbvi._iterations,    cbbvi._learning_rate,
                                                 cbbvi._record_elbo,   cbbvi._quiet_progress} {
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(cbbvi._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(cbbvi._optim)));
    */
}

CBBVI::CBBVI(CBBVI&& cbbvi) noexcept : BBVI(std::move(cbbvi)) {
    _log_p_blanket       = cbbvi._log_p_blanket;
    cbbvi._log_p_blanket = {};
}

CBBVI& CBBVI::operator=(const CBBVI& cbbvi) {
    if (this == &cbbvi)
        return *this;
    _neg_posterior   = cbbvi._neg_posterior;
    _q               = cbbvi._q;
    _sims            = cbbvi._sims;
    _approx_param_no = cbbvi._approx_param_no;
    _printer         = cbbvi._printer;
    _iterations      = cbbvi._iterations;
    _record_elbo     = cbbvi._record_elbo;
    _quiet_progress  = cbbvi._quiet_progress;
    _optimizer       = cbbvi._optimizer;
    _learning_rate   = cbbvi._learning_rate;
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(cbbvi._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(cbbvi._optim)));
    */
    _log_p_blanket = cbbvi._log_p_blanket;
    return *this;
}

CBBVI& CBBVI::operator=(CBBVI&& cbbvi) noexcept {
    _neg_posterior   = cbbvi._neg_posterior;
    _q               = cbbvi._q;
    _sims            = cbbvi._sims;
    _approx_param_no = cbbvi._approx_param_no;
    _printer         = cbbvi._printer;
    _iterations      = cbbvi._iterations;
    _record_elbo     = cbbvi._record_elbo;
    _quiet_progress  = cbbvi._quiet_progress;
    _optimizer       = cbbvi._optimizer;
    _learning_rate   = cbbvi._learning_rate;
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(cbbvi._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(cbbvi._optim)));
    */
    _log_p_blanket       = cbbvi._log_p_blanket;
    cbbvi._neg_posterior = {};
    cbbvi._q.resize(0);
    cbbvi._sims = 0;
    cbbvi._approx_param_no.resize(0, 0);
    cbbvi._printer        = false;
    cbbvi._iterations     = 0;
    cbbvi._record_elbo    = false;
    cbbvi._quiet_progress = true;
    cbbvi._optimizer      = nullptr;
    cbbvi._learning_rate  = 0;
    cbbvi._log_p_blanket  = {};
    return *this;
}

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
             std::function<double(Eigen::VectorXd)> full_neg_posterior, const std::vector<Normal>& q, int sims,
             std::string optimizer, int iterations, double learning_rate, int mini_batch, bool record_elbo,
             bool quiet_progress)
    : BBVI{std::function<double(Eigen::VectorXd)>(),
           q,
           sims,
           std::move(optimizer),
           iterations,
           learning_rate,
           record_elbo,
           quiet_progress},
      _neg_posterior{std::move(neg_posterior)}, _full_neg_posterior{std::move(full_neg_posterior)},
      _mini_batch{mini_batch} {}

BBVIM::BBVIM(const BBVIM& bbvim)
    : BBVIM{bbvim._neg_posterior, bbvim._full_neg_posterior, bbvim._q,
            bbvim._sims,          bbvim._optimizer,          bbvim._iterations,
            bbvim._learning_rate, bbvim._mini_batch,         bbvim._record_elbo,
            bbvim._quiet_progress} {}

BBVIM::BBVIM(BBVIM&& bbvim) noexcept : BBVI(std::move(bbvim)) {
    _mini_batch               = bbvim._mini_batch;
    _full_neg_posterior       = bbvim._full_neg_posterior;
    bbvim._mini_batch         = 0;
    bbvim._full_neg_posterior = {};
}

BBVIM& BBVIM::operator=(const BBVIM& bbvim) {
    if (this == &bbvim)
        return *this;
    _neg_posterior   = bbvim._neg_posterior;
    _q               = bbvim._q;
    _sims            = bbvim._sims;
    _approx_param_no = bbvim._approx_param_no;
    _printer         = bbvim._printer;
    _iterations      = bbvim._iterations;
    _record_elbo     = bbvim._record_elbo;
    _quiet_progress  = bbvim._quiet_progress;
    _optimizer       = bbvim._optimizer;
    _learning_rate   = bbvim._learning_rate;
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(bbvim._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(bbvim._optim)));
    */
    _full_neg_posterior = bbvim._full_neg_posterior;
    _mini_batch         = bbvim._mini_batch;
    return *this;
}

BBVIM& BBVIM::operator=(BBVIM&& bbvim) noexcept {
    _neg_posterior   = bbvim._neg_posterior;
    _q               = bbvim._q;
    _sims            = bbvim._sims;
    _approx_param_no = bbvim._approx_param_no;
    _printer         = bbvim._printer;
    _iterations      = bbvim._iterations;
    _record_elbo     = bbvim._record_elbo;
    _quiet_progress  = bbvim._quiet_progress;
    _optimizer       = bbvim._optimizer;
    _learning_rate   = bbvim._learning_rate;
    /*
    if (_optimizer == "ADAM")
        _optim = std::make_unique<ADAM>(dynamic_cast<ADAM&>(*(bbvim._optim)));
    else if (_optimizer == "RMSProp")
        _optim = std::make_unique<RMSProp>(dynamic_cast<RMSProp&>(*(bbvim._optim)));
    */
    _full_neg_posterior  = bbvim._full_neg_posterior;
    _mini_batch          = bbvim._mini_batch;
    bbvim._neg_posterior = {};
    bbvim._q.resize(0);
    bbvim._sims = 0;
    bbvim._approx_param_no.resize(0, 0);
    bbvim._printer            = false;
    bbvim._iterations         = 0;
    bbvim._record_elbo        = false;
    bbvim._quiet_progress     = true;
    bbvim._optimizer          = nullptr;
    bbvim._learning_rate      = 0;
    bbvim._full_neg_posterior = {};
    bbvim._mini_batch         = 0;
    return *this;
}

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

BBVIReturnData BBVIM::run(bool store) {
    return run_with(store, _full_neg_posterior);
}