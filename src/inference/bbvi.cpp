#include "inference/bbvi.hpp"

#include "multivariate_normal.hpp"

BBVI::BBVI(
        std::function<double(Eigen::VectorXd)> neg_posterior,
        std::vector<Family>& q,
        int sims,
        std::string optimizer = "RMSProp",
        int iterations = 1000,
        double learning_rate = 0.001,
        bool record_elbo = false,
        bool quiet_progress = false
        )
    : _neg_posterior{neg_posterior},
      _q{q},
      _sims{sims},
      _printer{true},
      _optimizer{optimizer},
      _iterations{iterations},
      _learning_rate{learning_rate},
      _record_elbo{record_elbo},
      _quiet_progress{quiet_progress}
{
    _approx_param_no = Eigen::VectorXd(_q.size());
    for (size_t i{0}; i < _q.size(); i++) {
        _approx_param_no[i] = _q[i].get_param_no();
    }
}

void BBVI::change_parameters(std::vector<double>& params) {
    short int no_of_params = 0;
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++) {
            _q[core_param].vi_change_param(approx_param, params[no_of_params]);
            no_of_params += 1;
        }
    }
}

double BBVI::create_normal_logq(Eigen::VectorXd& z) {
    auto means_scales = BBVI::get_means_and_scales();
    return Mvn::logpdf(z, means_scales.first, means_scales.second).sum();
}

std::vector<double> BBVI::current_parameters() {
    std::vector<double> current = std::vector<double>();
    for (size_t core_param = 0; core_param < _q.size(); core_param++) {
        for (size_t approx_param = 0; approx_param < _q[core_param].get_param_no(); approx_param++)
            current.push_back(_q[core_param].vi_return_param(approx_param));
    }
    return current;
};

double BBVI::cv_gradient(Eigen::VectorXd& z, bool initial) {
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(_approx_param_no.sum());
    Eigen::VectorXd z_t = z.transpose();
    if (initial)
        Eigen::VectorXd loq_q = normal_log_q(z_t);
    else
        igen::VectorXd loq_q = _initial(z_t);
    Eigen::VectorXd log_p = log_p(z_t);
    Eigen::VectorXd grad_log_q = grad_log_q(z);
    Eigen::VectorXd gradient = gradient.array() * (log_p - log_q).array();

    Eigen::VectorXd alpha0 = alpha_recursion(Eigen::VectorXd::Zero(_approx_param_no.sum()), grad_log_q, gradient, _approx_param_no.sum());

    double var = (grad_log_q.array() - grad_log_q.mean()).abs().exp(2).mean();
    Eigein::VectorXd vectorized = gradient - ((alpha0.array() / var) * grad_log_q.transpose().array()).transpose();

    return vectorized.mean();
}