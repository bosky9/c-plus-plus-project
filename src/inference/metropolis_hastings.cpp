/**
 * @file metropolis_hastings.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "inference/metropolis_hastings.hpp"

#include "inference/metropolis_sampler.hpp" // metropolis::metropolis_sampler()
#include "multivariate_normal.hpp"          // Mvn

#include <iostream> // std::cout

MetropolisHastings::MetropolisHastings(std::function<double(const Eigen::VectorXd&)>& posterior, double scale,
                                       int64_t nsims, const Eigen::VectorXd& initials,
                                       const std::optional<Eigen::MatrixXd>& cov_matrix, int64_t thinning,
                                       bool warm_up_period, bool quiet_progress)
    : _posterior{posterior}, _scale{scale}, _nsims{(1 + warm_up_period) * nsims * thinning}, _initials{initials},
      _param_no{initials.size()}, _thinning{thinning}, _warm_up_period{warm_up_period}, _quiet_progress{
                                                                                                quiet_progress} {

    _phi        = Eigen::MatrixXd::Zero(_nsims, _param_no);
    _phi.row(0) = _initials;

    _cov_matrix = cov_matrix.value_or(Eigen::MatrixXd::Identity(_param_no, _param_no));
    _cov_matrix.array().colwise() *= _initials.cwiseAbs().array();
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
    double acceptance{1.0};
    bool finish{true};

    while (acceptance < 0.234 || acceptance > 0.4 || finish) {
        // If acceptance is in range, proceed to sample, else continue tuning
        int64_t sims_to_do;
        if (!(acceptance < 0.234 || acceptance > 0.4)) {
            finish = false;
            if (!_quiet_progress) {
                std::cout << "\nTuning complete! Now sampling.";
            }
            sims_to_do = _nsims;
        } else {
            sims_to_do = _nsims / 2;
        }

        // Holds data on acceptance rates and uniform random numbers
        Eigen::VectorXd a_rate{Eigen::VectorXd::Zero(sims_to_do)};
        Eigen::VectorXd crit{Eigen::VectorXd::Random(sims_to_do)};
        Mvn post{Eigen::VectorXd::Zero(_param_no), _cov_matrix};
        Eigen::MatrixXd rnums(sims_to_do, _param_no);
        for (Eigen::Index i{0}; i < sims_to_do; ++i) {
            rnums.row(i) = post.sample() * _scale;
        }
        metropolis::metropolis_sampler(sims_to_do, _phi, _posterior, a_rate, rnums, crit);
        acceptance = a_rate.sum() / static_cast<double>(a_rate.size());
        _scale     = tune_scale(acceptance, _scale);
        if (!_quiet_progress) {
            std::cout << "Acceptance rate of Metropolis-Hastings is " << acceptance;
        }
    }

    // Remove warm-up and thin
    _phi = _phi(Eigen::seq(_nsims / 2, Eigen::last), Eigen::all)(Eigen::seq(0, Eigen::last, _thinning), Eigen::all);
    Eigen::MatrixXd chain = _phi.transpose();

    Eigen::VectorXd mean_est = _phi.colwise().mean();

    std::vector<double> median_est{}, upper_95_est{}, lower_5_est{};
    for (int64_t i{0}; i < _param_no; ++i) {
        Eigen::VectorXd col_sort{_phi.col(i)};
        std::sort(col_sort.begin(), col_sort.end());

        if (static_cast<bool>(_phi.rows() % 2))
            median_est.push_back(col_sort[_phi.rows() / 2]);
        else
            median_est.push_back((col_sort[_phi.rows() / 2] + col_sort[_phi.rows() / 2 - 1]) / 2);

        upper_95_est.push_back(col_sort[_phi.rows() * 95 / 100]);
        lower_5_est.push_back(col_sort[_phi.rows() * 5 / 100]);
    }

    return {chain, mean_est, Eigen::VectorXd::Map(&median_est[0], static_cast<Eigen::Index>(median_est.size())),
            Eigen::VectorXd::Map(&upper_95_est[0], static_cast<Eigen::Index>(upper_95_est.size())),
            Eigen::VectorXd::Map(&lower_5_est[0], static_cast<Eigen::Index>(lower_5_est.size()))};
}