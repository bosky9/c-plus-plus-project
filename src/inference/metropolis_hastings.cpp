/**
 * @file metropolis_hastings.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "inference/metropolis_hastings.hpp"

#include "inference/metropolis_sampler.hpp" // metropolis::metropolis_sampler()
#include "multivariate_normal.hpp"          // Mvn

#include <iostream> // std::cout

MetropolisHastings::MetropolisHastings(std::function<double(const Eigen::VectorXd&)>& posterior, double scale,
                                       size_t nsims, const Eigen::VectorXd& initials,
                                       const std::optional<Eigen::MatrixXd>& cov_matrix, size_t thinning,
                                       bool warm_up_period, bool quiet_progress)
    : _posterior{posterior}, _scale{scale}, _nsims{(1 + warm_up_period) * nsims * thinning}, _initials{initials},
      _param_no{static_cast<size_t>(initials.size())}, _thinning{thinning}, _warm_up_period{warm_up_period},
      _quiet_progress{quiet_progress} {

    _phi        = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(_nsims), static_cast<Eigen::Index>(_param_no));
    _phi.row(0) = _initials; //point from which to start the Metropolis-Hasting algorithm

    _cov_matrix = cov_matrix.value_or(
            Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(_param_no),
                                      static_cast<Eigen::Index>(_param_no)).array().colwise() *
                                      _initials.cwiseAbs().array()
            );
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

    while ((acceptance < 0.234 || acceptance > 0.4) || finish) {
        // If acceptance is in range, proceed to sample, else continue tuning
        size_t sims_to_do{_nsims};
        if (!(acceptance < 0.234 || acceptance > 0.4)) {
            finish = false;
            if (!_quiet_progress) {
                std::cout << "\nTuning complete! Now sampling.";
            }
        } else {
            sims_to_do = _nsims / 2;
        }

        // Holds data on acceptance rates and uniform random numbers
        Eigen::VectorXd a_rate{Eigen::VectorXd::Zero(static_cast<Eigen::Index>(sims_to_do))};
        // Numbers are uniformly spread through their whole definition range for integer types,
        // and in the [-1:1] range for floating point scalar types.
        Eigen::VectorXd crit{Eigen::VectorXd::Random(static_cast<Eigen::Index>(sims_to_do)).cwiseAbs()};
        Mvn post{Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_param_no)), _cov_matrix};
        Eigen::MatrixXd rnums(sims_to_do, _param_no);
        for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(sims_to_do); ++i) {
            rnums.row(i) = post.sample() * _scale;
        }
        metropolis::metropolis_sampler(sims_to_do, _phi, _posterior, a_rate, rnums, crit);
        acceptance = a_rate.sum() / static_cast<double>(a_rate.size());
        _scale     = tune_scale(acceptance, _scale);
        if (!_quiet_progress) {
            std::cout << "\nAcceptance rate of Metropolis-Hastings is " << acceptance;
        }
    }

    // Remove warm-up and thin
    Eigen::MatrixXd new_phi{_phi(Eigen::seq(_nsims / 2, Eigen::last), Eigen::all)(Eigen::seq(0, Eigen::last, _thinning), Eigen::all)};
    _phi = new_phi;
    Eigen::MatrixXd chain = _phi.transpose();
    chain = chain(Eigen::seq(0, _param_no-1), Eigen::all);

    Eigen::VectorXd mean_est = _phi(Eigen::seq(0, _param_no-1), Eigen::all).colwise().mean();

    std::vector<double> median_est{}, upper_95_est{}, lower_5_est{};
    for (Eigen::Index i{0}; i < static_cast<Eigen::Index>(_param_no); ++i) {
        Eigen::VectorXd col_sort{_phi.col(i)};
        std::sort(col_sort.begin(), col_sort.end());

        if (static_cast<bool>(_phi.rows() % 2))
            median_est.push_back(col_sort[_phi.rows() / 2]);
        else
            median_est.push_back((col_sort[_phi.rows() / 2] + col_sort[_phi.rows() / 2 - 1]) / 2);

        upper_95_est.push_back(col_sort[_phi.rows() * 95 / 100]);
        lower_5_est.push_back(col_sort[_phi.rows() * 5 / 100]);
    }

    return {chain, mean_est, Eigen::VectorXd::Map(median_est.data(), static_cast<Eigen::Index>(median_est.size())),
            Eigen::VectorXd::Map(upper_95_est.data(), static_cast<Eigen::Index>(upper_95_est.size())),
            Eigen::VectorXd::Map(lower_5_est.data(), static_cast<Eigen::Index>(lower_5_est.size()))};
}