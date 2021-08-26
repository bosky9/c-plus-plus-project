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

double BBVI::create_normal_logq(Eigen::VectorXd& z) {
    auto means_scales = BBVI::get_means_and_scales();
    return Mvn::logpdf(z, means_scales.first, means_scales.second).sum();
}

Eigen::VectorXd BBVI::cv_gradient(Eigen::MatrixXd& z, bool initial) {
    Eigen::VectorXd gradient   = Eigen::VectorXd::Zero(_approx_param_no.sum());
    Eigen::MatrixXd z_t        = z.transpose();
    Eigen::VectorXd log_q_res      = normal_log_q(z_t, initial);
    Eigen::VectorXd log_p_res      = log_p(z_t);
    Eigen::MatrixXd grad_log_q_res = grad_log_q(z);
    gradient                   = grad_log_q_res * (log_p_res - log_q_res);

    Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(_approx_param_no.sum());
    alpha_recursion(alpha0, grad_log_q_res, gradient, _approx_param_no.sum());

    double var = pow((grad_log_q_res.array() - grad_log_q_res.mean()).abs(), 2).mean();
    Eigen::VectorXd vectorized = gradient - ((alpha0 / var) * grad_log_q_res.transpose()).transpose();

    return vectorized.colwise().mean();
}

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


Eigen::MatrixXd BBVI::grad_log_q(Eigen::MatrixXd& z) {
    size_t param_count   = 0;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(_approx_param_no.sum(), _sims);
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++) {
            Eigen::VectorXd temp_z = z.row(core_param);
            grad.row(param_count) = _q[core_param].vi_score(temp_z, approx_param);
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
        if (i == round(_iterations/10*split)-1) {
            double post = -_neg_posterior(current_params);
            double approx = create_normal_logq(current_params);
            double diff = post - approx;
            if (! _quiet_progress) {
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
    Eigen::MatrixXd z = draw_normal(true);
    Eigen::VectorXd gradient = cv_gradient(z, true);
    //gradient[np.isnan(gradient)] = 0;
    //double variance = pow(gradient, 2);
    //Eigen::VectorXd final_parameters = _current_parameters();
    //size_t final_samples = 1;
}

BBVIReturnData BBVI::run_and_store() {}
