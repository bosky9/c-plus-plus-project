#include "inference/metropolis_hastings.hpp"

#include "multivariate_normal.hpp"
#include "inference/metropolis_sampler.hpp"

MetropolisHastings::MetropolisHastings(std::function<double(Eigen::VectorXd)>& posterior, double scale, int nsims,
                                       const Eigen::VectorXd& initials,
                                       const std::optional<Eigen::MatrixXd>& cov_matrix, int thinning,
                                       bool warm_up_period, // TODO: TSM model_object = nullptr,
                                       bool quiet_progress)
    : _posterior{posterior}, _scale{scale}, _nsims{(1 + warm_up_period) * nsims * thinning}, _initials{initials},
      _param_no{initials.size()}, _thinning{thinning}, _warm_up_period{warm_up_period}, _quiet_progress{
                                                                                                quiet_progress} {

    _phi        = Eigen::MatrixXd::Zero(_nsims, _param_no);
    _phi.row(0) = _initials;
    _cov_matrix = cov_matrix.value_or(Eigen::MatrixXd::Identity(_param_no, _param_no) * _initials.cwiseAbs());

    // if (model_object == nullptr)
    //  _model = model_object;
}

double MetropolisHastings::tune_scale(double acceptance, double scale) {
    if (acceptance > 0.8)
        scale *= 2.0;
    else if (acceptance <= 0.8 && acceptance > 0.4)
        scale *= 1.3;
    else if (acceptance < 0.234 && acceptance > 0.1)
        scale *= (1 / 1.3);
    else if (acceptance <= 0.1 && acceptance > 0.05)
        scale *= 0.4;
    else if (acceptance <= 0.05 && acceptance > 0.01)
        scale *= 0.2;
    else if (acceptance <= 0.01)
        scale *= 0.1;
    return scale;
}

Sample MetropolisHastings::sample() {
    double acceptance = 1.0;
    bool finish = true;

    while ((acceptance < 0.234 or acceptance > 0.4) or finish) {
        // If acceptance is in range, proceed to sample, else continue tuning
        int sims_to_do;
        if (!(acceptance < 0.234 or acceptance > 0.4)) {
                finish = false;
                if (!_quiet_progress) {
                    std::cout << "\nTuning complete! Now sampling.";
                }
                sims_to_do = _nsims;
        } else {
            sims_to_do = _nsims/2;
        }
        // Holds data on acceptance rates and uniform random numbers
        Eigen::VectorXd a_rate{Eigen::VectorXd::Zero(sims_to_do)};
        Eigen::VectorXd crit{Eigen::VectorXd::Random(sims_to_do)};
        Mvn post{Eigen::VectorXd::Zero(_param_no), _cov_matrix};
        Eigen::MatrixXd rnums;
        for (size_t i{0}; i < sims_to_do; i++) {
            rnums.row(i) = post.sample() * _scale;
        }
        metropolis_sampler(sims_to_do, _phi, _posterior, a_rate, rnums, crit);
        acceptance = a_rate.sum()/a_rate.size();
        _scale = tune_scale(acceptance, _scale);
        if (!_quiet_progress) {
            std::cout << "Acceptance rate of Metropolis-Hastings is " << acceptance;
        }
    }

    // Remove warm-up and thin
    _phi = _phi(Eigen::seq(_nsims/2, Eigen::last), Eigen::all)(Eigen::seq(0,Eigen::last,_thinning), Eigen::all);
    // TODO: controlla, transpose() non deve modificare _phi (anche in norm_post_sim.hpp)
    // Non lo fa, quella è la funzione transposeInPlace()
    Eigen::MatrixXd chain = _phi.transpose();

    std::vector<double> mean_est(_phi.cols());
    Eigen::VectorXd mean_vector = _phi.colwise().mean();
    mean_est.resize(mean_vector.size());
    Eigen::VectorXd::Map(&mean_est[0], mean_vector.size()) = mean_vector;

    std::vector<double> median_est;
    std::vector<double> upper_95_est;
    std::vector<double> lower_5_est;
    for (size_t i{0}; i < _param_no; i++) {
        std::vector<double> col_sort(_phi.rows());
        Eigen::VectorXd::Map(&col_sort[0], _nsims) = _phi.col(i);
        std::sort(col_sort.begin(), col_sort.end());

        if (static_cast<bool>(_phi.rows() % 2))
            median_est.push_back(col_sort[_phi.rows() / 2]);
        else
            median_est.push_back((col_sort[_phi.rows() / 2] + col_sort[_phi.rows() / 2 - 1]) / 2);

        upper_95_est.push_back((col_sort[_phi.rows() * 95 / 100]));
        lower_5_est.push_back((col_sort[_phi.rows() * 5 / 100]));
    }

    return {chain, mean_est, median_est, upper_95_est, lower_5_est};
}