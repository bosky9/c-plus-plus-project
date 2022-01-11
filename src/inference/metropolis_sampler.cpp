/**
 * @file metropolis_sampler.cpp
 * @author Bodini Alessia, Boschi Federico, Cinquetti Ettore
 * @date January, 2022
 */

#include "inference/metropolis_sampler.hpp"

void metropolis::metropolis_sampler(size_t sims_to_do, Eigen::MatrixXd& phi,
                                    const std::function<double(const Eigen::VectorXd&)>& posterior,
                                    Eigen::VectorXd& a_rate, const Eigen::MatrixXd& rnums,
                                    const Eigen::VectorXd& crit) {

    double old_lik = -posterior(phi.row(0)); // float in Cython

    Eigen::VectorXd phi_prop;
    for (Eigen::Index i{1}; i < static_cast<Eigen::Index>(sims_to_do); ++i) {
        phi_prop         = phi.row(i - 1) + rnums.row(i);
        double post_prop = -posterior(phi_prop);
        double lik_rate  = exp(post_prop - old_lik);

        if (crit[i] < lik_rate) {
            phi.row(i) = phi_prop;
            a_rate[i]  = 1;
            old_lik    = post_prop;
        } else
            phi.row(i) = phi.row(i - 1);
    }
}