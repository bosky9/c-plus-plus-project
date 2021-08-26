#include "inference/bbvi.hpp"

#include "inference/bbvi_routines.hpp"
#include "multivariate_normal.hpp"

BBVI::BBVI(std::function<double(Eigen::VectorXd)> neg_posterior, std::vector<Normal>& q, int sims,
           std::string optimizer, int iterations, double learning_rate, bool record_elbo, bool quiet_progress)
    : _neg_posterior{neg_posterior}, _q{q}, _sims{sims}, _printer{true}, _optimizer{optimizer}, _iterations{iterations},
      _learning_rate{learning_rate}, _record_elbo{record_elbo}, _quiet_progress{quiet_progress} {
    _approx_param_no = Eigen::VectorXd(_q.size());
    for (size_t i{0}; i < _q.size(); i++) {
        _approx_param_no[i] = _q[i].get_param_no();
    }
}

BBVI::~BBVI() {
    delete _optim;
}

void BBVI::change_parameters(std::vector<double>& params) {
    size_t no_of_params = 0;
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++) {
            _q[core_param].vi_change_param(approx_param, params[no_of_params]);
            no_of_params++;
        }
    }
}

double BBVI::create_normal_logq(Eigen::MatrixXd& z) {
    auto means_scales = BBVI::get_means_and_scales();
    return Mvn::logpdf(z, means_scales.first, means_scales.second).sum();
}

/*
double BBVI::cv_gradient(Eigen::VectorXd& z, bool initial) {
    Eigen::VectorXd gradient   = Eigen::VectorXd::Zero(_approx_param_no.sum());
    Eigen::VectorXd z_t        = z.transpose();
    Eigen::VectorXd log_q      = normal_log_q(z_t, initial);
    Eigen::VectorXd log_p      = log_p(z_t);
    Eigen::MatrixXd grad_log_q = grad_log_q(z);
    gradient                   = gradient.array() * (log_p - log_q).array();

    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(_approx_param_no.sum());
    alpha_recursion(alpha0, grad_log_q, gradient, _approx_param_no.sum());

    double var = pow((grad_log_q.array() - grad_log_q.mean()).abs(), 2).mean();
    Eigen::VectorXd vectorized =
            gradient -
            static_cast<Eigen::VectorXd>((alpha0.array() / var) * grad_log_q.transpose().array()).transpose();

    return vectorized.mean();
}
 */

std::vector<double> BBVI::current_parameters() {
    std::vector<double> current = std::vector<double>();
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++)
            current.push_back(_q[core_param].vi_return_param(approx_param));
    }
    return current;
};

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
    Eigen::VectorXd means = Eigen::VectorXd::Zero(_q.size());
    Eigen::VectorXd scale = Eigen::VectorXd::Zero(_q.size());

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

Eigen::MatrixXd BBVI::grad_log_q(Eigen::VectorXd& z) {
    size_t param_count   = 0;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(_approx_param_no.sum(), _sims);
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++) {
            grad(param_count) = _q[core_param].vi_score(z[core_param], approx_param);
            param_count++;
        }
    }
}


std::vector<Normal> BBVI::get_q() const {
    return _q;
}

double BBVI::cv_gradient(Eigen::VectorXd& z, bool initial) {}
Eigen::VectorXd BBVI::log_p(Eigen::VectorXd& z) {}
Eigen::VectorXd BBVI::normal_log_q(Eigen::VectorXd& z, bool initial) {}
void BBVI::print_progress(double i, Eigen::VectorXd& current_params) {}
double BBVI::get_elbo(Eigen::VectorXd& current_params) {}
BBVIReturnData BBVI::run() {}
BBVIReturnData BBVI::run_and_store() {}

/*
Eigen::VectorXd BBVI::log_p(Eigen::MatrixXd& z) {
    return log_p_posterior(z, _neg_posterior);
}

*/